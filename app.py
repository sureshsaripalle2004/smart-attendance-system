import streamlit as st
import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

DATASET = "dataset"
ATT_FILE = "attendance.csv"

os.makedirs(DATASET, exist_ok=True)

students_df = pd.read_csv("students.csv")

def get_student_details(name):
    row = students_df[students_df["Name"] == name]
    if not row.empty:
        return row.iloc[0]["Roll"], row.iloc[0]["Admission"]
    return "N/A", "N/A"

@st.cache_resource
def load_faces():
    encodings = []
    names = []
    for person in os.listdir(DATASET):
        person_path = os.path.join(DATASET, person)
        if not os.path.isdir(person_path):
            continue
        for img in os.listdir(person_path):
            path = os.path.join(person_path, img)
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)
            if len(enc) > 0:
                encodings.append(enc[0])
                names.append(person)
    return encodings, names

def mark_attendance(name):
    if not os.path.exists(ATT_FILE):
        df = pd.DataFrame(columns=["Name","Roll","Admission","Time"])
        df.to_csv(ATT_FILE, index=False)
    df = pd.read_csv(ATT_FILE)
    if name not in df["Name"].values:
        roll, admission = get_student_details(name)
        now = datetime.now()
        df.loc[len(df)] = [name, roll, admission, now.strftime("%H:%M:%S")]
        df.to_csv(ATT_FILE, index=False)

def process_image(frame, known_enc, known_names):
    rgb = frame[:, :, ::-1]
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)
    for (top, right, bottom, left), enc in zip(faces, encodings):
        matches = face_recognition.compare_faces(known_enc, enc)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
            mark_attendance(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame

st.title("📱 Smart Attendance System")

menu = st.sidebar.selectbox("Menu", [
    "Register Student",
    "Take Attendance",
    "Dashboard"
])

if menu == "Register Student":
    name = st.text_input("Enter Name")
    img_file = st.camera_input("Take Photo")
    if img_file and name:
        path = f"{DATASET}/{name}"
        os.makedirs(path, exist_ok=True)
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        cv2.imwrite(f"{path}/0.jpg", img)
        st.success("Registered Successfully!")

elif menu == "Take Attendance":
    uploaded = st.file_uploader("Upload Classroom Image", type=["jpg","png"])
    if uploaded:
        known_enc, known_names = load_faces()
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = process_image(frame, known_enc, known_names)
        st.image(frame, channels="BGR")

elif menu == "Dashboard":
    if os.path.exists(ATT_FILE):
        df = pd.read_csv(ATT_FILE)
        st.dataframe(df)
        st.bar_chart(df["Name"].value_counts())
    else:
        st.warning("No data yet")
