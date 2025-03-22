import cv2
import face_recognition
import threading
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from collections import deque, Counter

# Allow the user to choose a model at runtime
print("Select an emotion detection model:")
print("1 - trpakov/vit-face-expression")
print("2 - dima806/facial_emotions_image_detection")
print("3 - motheecreator/vit-Facial-Expression-Recognition")
choice = input("Enter model number: ").strip()

# Load the chosen model and processor
if choice == "2":
    model_name = "dima806/facial_emotions_image_detection"
elif choice == "3":
    model_name = "motheecreator/vit-Facial-Expression-Recognition"
else:
    model_name = "trpakov/vit-face-expression"  # Default model

processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Select device (using GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device
model.to(device)

# Global variables for threading
face_locations = []    # Shared variable with detected face coordinates
global_frame = None    # Global frame for the detection thread to process
frame_lock = threading.Lock()  # Lock to safely update/access global_frame

# Initialize webcam
video_capture = cv2.VideoCapture(0)
frame_count = 0

# Emotion history (to track the last 100 detected emotions)
emotion_history = deque(maxlen=100)
all_emotions = list(model.config.id2label.values())

def get_emotion_distribution():
    counter = Counter(emotion_history)
    total = sum(counter.values())
    return {emotion: (counter[emotion] / total * 100) if total > 0 else 0 for emotion in all_emotions}

# Thread function to continuously detect faces on a downscaled frame
def process_frame():
    global global_frame, face_locations
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            # Work on a copy to avoid race conditions
            frame_copy = global_frame.copy()
        # Downscale for faster processing (50% of original size)
        small_frame = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Detect faces using the HOG-based model
        small_face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        # Scale the detected coordinates back to the original frame size
        face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in small_face_locations]

# Start the face detection thread
detection_thread = threading.Thread(target=process_frame, daemon=True)
detection_thread.start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    # Update the global frame for the detection thread every 3rd frame
    if frame_count % 3 == 0:
        with frame_lock:
            global_frame = frame.copy()

    # Make a local copy of face_locations to prevent race conditions
    current_face_locations = face_locations.copy()
    faces = []
    # Extract each face region for emotion recognition
    for (top, right, bottom, left) in current_face_locations:
        face_image = frame[top:bottom, left:right]
        # Resize face image to the expected input size for the model (224x224)
        try:
            face_resized = cv2.resize(face_image, (224, 224))
        except Exception:
            continue  # Skip faces that may be too small or cause errors
        # Convert to PIL Image (and to RGB) for the processor
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        faces.append(face_pil)
    
    # If faces are detected, run emotion recognition
    if faces:
        inputs = processor(images=faces, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1).cpu().numpy()
        emotions = [model.config.id2label[pred] for pred in predictions]
        
        # Annotate each detected face with a bounding box and the emotion label
        for (top, right, bottom, left), emotion in zip(current_face_locations, emotions):
            emotion_history.append(emotion)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display emotion distribution on the frame
    distribution = get_emotion_distribution()
    y_offset = 20
    cv2.putText(frame, "Mood", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for emotion, percentage in distribution.items():
        y_offset += 30
        text = f"{percentage:.0f}% {emotion}"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Optimized Face Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
