import face_recognition
import cv2
import os
import pandas as pd
from datetime import datetime
from tkinter import *
from PIL import Image, ImageTk
import joblib
# Load your trained classifier (from the training phase)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
# Load the trained model
# Load your trained classifier (from the training phase)
from sklearn.neighbors import KNeighborsClassifier
import joblib

classifier = joblib.load('trained_model.pkl')  # Load the trained model using joblib
 # Use the path where you saved your trained model

# Rest of the testing code...
# ... [Capture image, perform recognition, mark attendance, etc.] ...

# Initialize or read existing attendance DataFrame from Excel
attendance_filename = 'attendance.xlsx'
if not os.path.isfile(attendance_filename):
    attendance_df = pd.DataFrame(columns=['Name', 'Time'])
else:
    attendance_df = pd.read_excel(attendance_filename)

# Function to capture image, recognize faces, mark attendance, and save to Excel
def capture_and_mark_attendance():
    global attendance_df  # Declare as global to indicate that it's the same variable from the outer scope

    video_capture = cv2.VideoCapture(0)  # Webcam capture
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to capture image
            break

    # Capture image and perform face recognition
    cv2.imwrite("temp_image.jpg", frame)
    test_image = face_recognition.load_image_file("temp_image.jpg")
    face_locations = face_recognition.face_locations(test_image)

    if len(face_locations) > 0:
        test_encoding = face_recognition.face_encodings(test_image)[0]  # Assuming one face in the captured image

        # Perform face recognition on the captured image
        prediction = classifier.predict([test_encoding])

        # Display the predicted name
        predicted_name = prediction[0]
        print(f"The predicted person is: {predicted_name}")

        # Mark attendance and save to Excel
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if predicted_name not in attendance_df['Name'].values:
            new_row = pd.DataFrame({'Name': [predicted_name], 'Time': [current_time]})
            attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
            attendance_df.to_excel(attendance_filename, index=False)
        else:
            print("Person already marked attendance.")
        


    video_capture.release()
    cv2.destroyAllWindows()

# GUI
root = Tk()
root.title("Face Recognition Attendance System")

# Button to start capturing and marking attendance
capture_button = Button(root, text="Capture and Mark Attendance", command=capture_and_mark_attendance)
capture_button.pack()

root.mainloop()
