import cv2
import os
import numpy as np
import face_recognition as fr
from tkinter import messagebox
import tkinter as tk


def encode_known_faces():
    # encodes faces
    encoded_faces = {}
    for dirpath, dnames, fnames in os.walk('faces'):
        for f in fnames:
            if f.endswith('.jpeg') or f.endswith('png'):
                face = fr.load_image_file('faces/' + f)
                encoding = fr.face_encodings(face)[0]
                encoded_faces[f.split('.')[0]] = encoding

    return encoded_faces


def unknown_face(img):
    # this function encodes a face based on file name
    face = fr.load_image_file('faces' + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(image):
    # classifies face
    faces = encode_known_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(image, 1)
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for faces_encoding in unknown_face_encodings:
        matches = fr.compare_faces(faces_encoded, faces_encoding)
        name = "Unknown"

        face_distances = fr.face_distance(faces_encoded, faces_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # draw box around face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

            # label for face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            cv2.putText(img, name, (left - 20, top - 20), font, 1.0, (255, 255, 255), 2)

    # Display image with facial recognition
    count = 0

    while True:
        if count == 0:
            root = tk.Tk()
            root.configure(background="#74D5DD")
            tk.Label(root, text="Facial Recognition", font=('Times', 20, 'bold'), bg="#74D5DD").pack()
            tk.Label(root, text="Press 'q' to quit", font=('Times', 14, 'bold'), bg="#74D5DD").pack()
            count += 1
            messagebox.showinfo("Detected", f"Faces detected: {face_names}")
            root.destroy()
        cv2.imshow('Facial Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.system(f"say Faces detected: {' '.join(face_names)}")
            return f"[DONE] Faces detected: {face_names}"


print(classify_face("test.jpeg"))
