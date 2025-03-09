import cv2
import face_recognition
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load the model and processor
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Use GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

model.to(device)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convert frame to RGB (face_recognition uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    for top, right, bottom, left in face_locations:
        # Extract the face from the frame
        face_image = rgb_frame[top:bottom, left:right]

        # Convert face to PIL format
        face_pil = Image.fromarray(face_image)

        inputs = processor(images=face_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}


        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted emotion
        predicted_label = outputs.logits.argmax(-1).item()
        emotion = model.config.id2label[predicted_label]

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display emotion label above the face
        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Emotion Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
