import face_recognition
import os
import joblib

# Path to the folder containing face images for training
data_path = r"D:\attendance using face recognation\train_data"
known_encodings = []
known_names = []

# Iterate through each image file for training
for image_file in os.listdir(data_path):
    image = face_recognition.load_image_file(os.path.join(data_path, image_file))
    
    # Check if a face is detected in the image
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        encoding = face_encodings[0]  # Assuming one face per image
        # Use file name (without extension) as the person's name
        name = os.path.splitext(image_file)[0]
        known_encodings.append(encoding)
        known_names.append(name)  # Use file name as person's name

# Train your model (using a simple k-Nearest Neighbors classifier)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(known_encodings, known_names)

# Save the trained model
joblib.dump(classifier, 'trained_model.pkl')
