import cv2
import face_recognition
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from collections import deque, Counter

# Load model and processor
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

video_capture = cv2.VideoCapture(0)
frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Store last 100 detected emotions
emotion_history = deque(maxlen=100)
all_emotions = list(model.config.id2label.values())

def get_emotion_distribution():
    counter = Counter(emotion_history)
    total = sum(counter.values())
    return {emotion: (counter[emotion] / total * 100) if total > 0 else 0 for emotion in all_emotions}

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    faces = []
    for top, right, bottom, left in face_locations:
        face_image = rgb_frame[top:bottom, left:right]
        face_pil = Image.fromarray(cv2.resize(face_image, (224, 224)))
        faces.append(face_pil)
    
    if faces:
        inputs = processor(images=faces, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = outputs.logits.argmax(-1).cpu().numpy()
        emotions = [model.config.id2label[pred] for pred in predictions]
        
        for (top, right, bottom, left), emotion in zip(face_locations, emotions):
            emotion_history.append(emotion)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display emotion distribution
    distribution = get_emotion_distribution()
    y_offset = 20
    cv2.putText(frame, "Mood", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for emotion, percentage in distribution.items():
        y_offset += 30
        text = f"{percentage:.0f}% {emotion}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Face Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
