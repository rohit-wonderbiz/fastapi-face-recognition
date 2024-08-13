import cv2
import os
import numpy as np
import pickle
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config import cameraType, waitTime, connection_string
import face_recognition
import pyodbc
from mtcnn import MTCNN
import random
from fastapi import Form, UploadFile, File, HTTPException

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize MTCNN
mtcnn_detector = MTCNN()

# Directory to save unknown faces
unknown_faces_dir = "unknown_faces"
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# List to track recently detected unknown faces
recent_unknown_faces = []

last_attendance_time = {}
# Directory for storing images
IMAGES_PATH = 'images/'

# Your existing database connection
conn = pyodbc.connect(connection_string)

recent_unknown_faces = []
recent_detection_interval = 30  # seconds
def is_recently_detected(face_encoding):
    current_time = time.time()
    for recent_face in recent_unknown_faces:
        recent_encoding, recent_time = recent_face
        if face_recognition.compare_faces([recent_encoding], face_encoding, tolerance=0.3)[0]:
            if current_time - recent_time < recent_detection_interval:
                return True
    return False

def update_recent_unknown_faces(face_encoding):
    current_time = time.time()
    recent_unknown_faces.append((face_encoding, current_time))
    # Keep only recent detections within the time interval
    recent_unknown_faces[:] = [face for face in recent_unknown_faces if current_time - face[1] < recent_detection_interval]


# Load known encodings from database
def load_encodings_from_db(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT UserId, FirstName, FaceEncoding FROM EmployeeDetails WHERE DATALENGTH(FaceEncoding) > DATALENGTH(0x)')
    rows = cursor.fetchall()
    return [row[0] for row in rows], [row[1] for row in rows], [pickle.loads(row[2]) for row in rows]
def detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn):
    def mark_attendance(conn, userId):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO AttendanceLogs (UserId, AttendanceLogTime, CheckType) VALUES (?, ?, ?)''',
            (userId, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), cameraType,))
        conn.commit()
        print(f"Marked Attendance for {userId}")

    
    # Detect faces using MTCNN
    detections = mtcnn_detector.detect_faces(frame)
    face_locations = []
    face_encodings = []

    for detection in detections:
        x, y, width, height = detection['box']
        x1, y1 = x + width, y + height

        # Extract the face ROI
        face = frame[y:y1, x:x1]
        
        # Convert face to RGB and encode it
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_face)

        if face_encoding:
            face_encodings.append(face_encoding[0])
            face_locations.append((y, x1, y1, x))

    face_names = []
    current_time = time.time()
    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index] and face_distances[best_match_index] < 0.45:
            id = known_face_id[best_match_index]
            name = known_face_names[best_match_index]
            if face_distances[best_match_index] < 0.35:
                if name not in last_attendance_time or (current_time - last_attendance_time[name]) > waitTime:
                    last_attendance_time[name] = current_time
                    mark_attendance(conn, id)
        else:
            # Check if the face was recently detected
            if not is_recently_detected(face_encoding):
                # Save the unknown face
                unknown_face_filename = os.path.join(unknown_faces_dir, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
                cv2.imwrite(unknown_face_filename, frame[face_locations[i][0]:face_locations[i][2], face_locations[i][3]:face_locations[i][1]])
                
                # Update the list of recently detected unknown faces
                update_recent_unknown_faces(face_encoding)
        
        face_names.append(name)

    return face_locations, face_names

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    # Load known face encodings
    known_face_id, known_face_names, known_face_encodings = load_encodings_from_db(conn)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_names = detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.png', frame)
        frame_data = buffer.tobytes()

        await websocket.send_bytes(frame_data)

    cap.release()

@app.get("/detect-employee/", response_class=HTMLResponse)
async def detect_employee():
    return templates.TemplateResponse("detect_employee.html", {"request": {}})

@app.post("/capture-face/")
async def capture_face(employee_id: str):
    person_dir = os.path.join('images', employee_id)
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if ret:
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1, 9999)}.jpg"
        img_path = os.path.join(person_dir, filename)
        cv2.imwrite(img_path, frame)
        cap.release()
        return {"status": "success", "image_path": img_path}
    else:
        cap.release()
        raise HTTPException(status_code=500, detail="Failed to capture image.")

@app.post("/save-encodings/")
async def save_encodings(employee_id: str):
    def process_and_save_encodings(img_paths, employee_id, conn):
        def save_encoding_to_db(conn, employee_id, encoding):
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE EmployeeDetails
                SET FaceEncoding = (?)
                WHERE UserId = (?)
            ''', (pickle.dumps(encoding), employee_id))
            conn.commit()

        encodings = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is not None:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if len(face_recognition.face_encodings(rgb_img)) > 0:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]
                    encodings.append(img_encoding)
                    os.remove(img_path)
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            save_encoding_to_db(conn, employee_id, avg_encoding)
            return {"status": "success", "message": f"Encodings for {employee_id} updated successfully!"}
        else:
            raise HTTPException(status_code=400, detail="No valid images to encode.")

    person_dir = os.path.join('images', employee_id)
    img_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith('.jpg')]

    if img_paths:
        return process_and_save_encodings(img_paths, employee_id, conn)
    else:
        raise HTTPException(status_code=400, detail="No images captured to save.")
    
@app.get("/", response_class=HTMLResponse)
async def check_employee_page():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/capture-faces", response_class=HTMLResponse)
async def capture_faces_page(employee_id: str, employee_name: str):
    return templates.TemplateResponse("capture_faces.html", {"request": {}, "employee_id": employee_id, "employee_name": employee_name})


@app.post("/check-employee/")
async def check_employee(employee_id: str = Form(...)):
    cursor = conn.cursor()
    cursor.execute('SELECT Id, FirstName FROM EmployeeDetails WHERE UserId = (?)', (employee_id,))
    result = cursor.fetchone()
    if result:
        return {"status": "success", "employee_id": employee_id, "employee_name": result[1]}
    else:
        raise HTTPException(status_code=404, detail="Employee ID not found!")


@app.post("/capture-image/")
async def capture_image(file: UploadFile = File(...), employee_id: str = Form(...)):
    person_dir = os.path.join(IMAGES_PATH, str(employee_id))
    os.makedirs(person_dir, exist_ok=True)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1,9999)}.jpg"
    img_path = os.path.join(person_dir, filename)
    cv2.imwrite(img_path, img)

    return {"status": "Image saved", "image_path": img_path}


@app.post("/save-encoding/")
async def save_encoding(employee_id: str = Form(...)):
    person_dir = os.path.join(IMAGES_PATH, str(employee_id))
    img_paths = [os.path.join(person_dir, fname) for fname in os.listdir(person_dir) if fname.endswith('.jpg')]

    if not img_paths:
        raise HTTPException(status_code=400, detail="No images found for encoding!")

    encodings = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_encodings = face_recognition.face_encodings(rgb_img)
            if img_encodings:
                encodings.append(img_encodings[0])
                os.remove(img_path)  # Optionally remove the image after processing

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE EmployeeDetails
            SET FaceEncoding = (?)
            WHERE UserId = (?)
        ''', (pickle.dumps(avg_encoding), employee_id))
        conn.commit()
        return {"status": "Encoding saved"}
    else:
        raise HTTPException(status_code=400, detail="No valid face encodings found!")
