# DLIB

# import cv2
# import numpy as np
# import pickle
# import time
# from datetime import datetime
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from config import cameraType, waitTime, server, port, password, user, database
# import face_recognition
# import pyodbc

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # Load your database connection
# connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server},{port};DATABASE={database};UID={user};PWD={password}'
# conn = pyodbc.connect(connection_string)

# def load_encodings_from_db(conn):
#     cursor = conn.cursor()
#     cursor.execute('SELECT UserId, FirstName, FaceEncoding FROM EmployeeDetails WHERE DATALENGTH(FaceEncoding) > DATALENGTH(0x)')
#     rows = cursor.fetchall()
#     return [row[0] for row in rows], [row[1] for row in rows], [pickle.loads(row[2]) for row in rows]

# def detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn):
#     def mark_attendance(conn, userId):
#         cursor = conn.cursor()
#         cursor.execute('''
#             INSERT INTO AttendanceLogs (UserId, AttendanceLogTime, CheckType) VALUES (?, ?, ?)''',
#             (userId, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), cameraType,))
#         conn.commit()
#         print(f"Marked Attendance for {userId}")

#     last_attendance_time = {}
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     small_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_small_frame)
#     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#     face_names = []
#     current_time = time.time()
#     for face_encoding in face_encodings:
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#         best_match_index = np.argmin(face_distances)
#         if matches[best_match_index] and face_distances[best_match_index] < 0.45:
#             id = known_face_id[best_match_index]
#             name = known_face_names[best_match_index]
#             if face_distances[best_match_index] < 0.3:
#                 if name not in last_attendance_time or (current_time - last_attendance_time[name]) > waitTime:
#                     last_attendance_time[name] = current_time
#                     mark_attendance(conn, id)
#         face_names.append(name)

#     return face_locations, face_names

# async def video_stream(websocket: WebSocket):
#     await websocket.accept()
#     cap = cv2.VideoCapture(0)  # Use the first camera

#     # Load known face encodings
#     known_face_id, known_face_names, known_face_encodings = load_encodings_from_db(conn)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect faces in the frame
#         face_locations, face_names = detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn)

#         # Draw bounding boxes and names on the frame
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             # Draw a label with a name below the face
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

#         # Encode frame to JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_data = buffer.tobytes()

#         await websocket.send_bytes(frame_data)

#     cap.release()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     try:
#         await video_stream(websocket)
#     except WebSocketDisconnect:
#         print("Client disconnected")

# @app.get("/", response_class=HTMLResponse)
# async def get():
#     return templates.TemplateResponse("index.html", {"request": {}})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


#MTCNN
import cv2
import os
import numpy as np
import pickle
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config import cameraType, waitTime, server, port, password, user, database
import face_recognition
import pyodbc
from mtcnn import MTCNN

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Disable TensorFlow verbose logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MTCNN
mtcnn_detector = MTCNN()

# Directory to save unknown faces
unknown_faces_dir = "unknown_faces"
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# List to track recently detected unknown faces
recent_unknown_faces = []

# Time interval to consider a face as recently detected (in seconds)
recent_detection_interval = 10

# Load your database connection
connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server},{port};DATABASE={database};UID={user};PWD={password}'
conn = pyodbc.connect(connection_string)

def load_encodings_from_db(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT UserId, FirstName, FaceEncoding FROM EmployeeDetails WHERE DATALENGTH(FaceEncoding) > DATALENGTH(0x)')
    rows = cursor.fetchall()
    return [row[0] for row in rows], [row[1] for row in rows], [pickle.loads(row[2]) for row in rows]

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

def detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn):
    def mark_attendance(conn, userId):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO AttendanceLogs (UserId, AttendanceLogTime, CheckType) VALUES (?, ?, ?)''',
            (userId, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), cameraType,))
        conn.commit()
        print(f"Marked Attendance for {userId}")

    last_attendance_time = {}
    
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
            if face_distances[best_match_index] < 0.3:
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

async def video_stream(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Use the first camera

    # Load known face encodings
    known_face_id, known_face_names, known_face_encodings = load_encodings_from_db(conn)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        face_locations, face_names = detect_known_faces(known_face_id, known_face_names, known_face_encodings, frame, conn)

        # Draw bounding boxes and names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.png', frame)
        frame_data = buffer.tobytes()

        await websocket.send_bytes(frame_data)

    cap.release()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await video_stream(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/", response_class=HTMLResponse)
async def get():
    return templates.TemplateResponse("index.html", {"request": {}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
