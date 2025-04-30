import face_recognition
import numpy as np
import os

# Paths
input_folder = "known_faces"
output_folder = "user_encodings"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            name = os.path.splitext(filename)[0]  # e.g., "anas.jpg" -> "anas"
            image_path = os.path.join(input_folder, filename)
            print(f"🔍 Processing {image_path}...")

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"❌ No face found in {filename}. Skipping.")
                continue

            encoding = encodings[0]
            np.save(os.path.join(output_folder, f"{name}.npy"), encoding)
            print(f"✅ Saved encoding: {name}.npy")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
