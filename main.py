from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load known faces
known_faces = {}
print("🔄 Loading known face encodings...")
for file in os.listdir("user_encodings"):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        known_faces[name] = np.load(f"user_encodings/{file}")
        print(f"✅ Loaded encoding for: {name}")
print("✅ All known encodings loaded.\n")

@app.post("/verify-face/")
async def verify_face(image: UploadFile = File(...)):
    print(f"📥 Received file: {image.filename} ({image.content_type})")

    try:
        # Read and convert image
        img_bytes = await image.read()
        print(f"📦 File size: {len(img_bytes)} bytes")

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Get face encodings
        encodings = face_recognition.face_encodings(img_np)
        print(f"🧠 Faces detected: {len(encodings)}")

        if not encodings:
            print("❌ No face detected in image.")
            return JSONResponse({"match": False, "reason": "No face detected"}, status_code=400)

        face_encoding = encodings[0]

        # Compare with known faces
        for name, known_encoding in known_faces.items():
            match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
            print(f"🔍 Comparing with {name} → Match: {match}")
            if match:
                print(f"✅ MATCH FOUND: {name}")
                return {"match": True, "user": name}

        print("❌ No match found.")
        return {"match": False}

    except Exception as e:
        print(f"🔥 ERROR: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
