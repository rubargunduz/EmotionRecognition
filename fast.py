import cv2
import face_recognition
import threading
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from collections import deque, Counter
import datetime
import time
import argparse
import platform
import subprocess

if platform.system() == 'Windows':
    import winsound
    def sound_alert():
        winsound.Beep(1000, 500)  # 1 kHz for 0.5 s
elif platform.system() == 'Darwin':
    def sound_alert():
        # play the default "Glass" sound
        subprocess.call(['afplay', '/System/Library/Sounds/Glass.aiff'])
else:
    def sound_alert():
        # on many Linux distros this will work if 'beep' is installed
        try:
            subprocess.call(['beep', '-f', '1000', '-l', '500'])
        except FileNotFoundError:
            # fallback to terminal BEL
            print('\a', end='', flush=True)

# set your alert threshold (in %)
TIREDNESS_ALERT_THRESHOLD = 75.0
# set your stress alert threshold (in %)
STRESS_ALERT_THRESHOLD = 75.0


parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['1','2','3','4'])
args = parser.parse_args()

if args.model:
    choice = args.model
else:
    print("Select an emotion detection model:")
    print("1 - trpakov/vit-face-expression")
    print("2 - HardlyHumans/Facial-expression-detection")
    print("3 - motheecreator/vit-Facial-Expression-Recognition")
    print("4 - Combined")
    choice = input("Enter model number: ").strip()

# Select device (using GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

all_emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Normalize emotion labels
label_map = {
    'angry': 'anger',
    'anger': 'anger',
    'fear': 'fear',
    'scared': 'fear',
    'sadness': 'sad',
    'sad': 'sad',
    'disgust': 'disgust',
    'happy': 'happy',
    'happiness': 'happy',
    'surprise': 'surprise',
    'neutral': 'neutral',
    'contempt': 'neutral'
}

# Load the chosen model and processor
if choice == "4":
    model_names = ["trpakov/vit-face-expression", "HardlyHumans/Facial-expression-detection", "motheecreator/vit-Facial-Expression-Recognition"]
    processors = [AutoImageProcessor.from_pretrained(name) for name in model_names]
    models = [ViTForImageClassification.from_pretrained(name).to(device) for name in model_names]

else:
    if choice == "2":
        model_name = "HardlyHumans/Facial-expression-detection"
    elif choice == "3":
        model_name = "motheecreator/vit-Facial-Expression-Recognition"
    else:
        model_name = "trpakov/vit-face-expression"  # Default model

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)

# Global variables for threading
face_locations = []    # Shared variable with detected face coordinates
global_frame = None    # Global frame for the detection thread to process
frame_lock = threading.Lock()  # Lock to safely update/access global_frame

# Initialize webcam
video_capture = cv2.VideoCapture(0)
frame_count = 0

# Emotion history (to track the last 100 detected emotions)
emotion_history = deque(maxlen=100)

def get_emotion_distribution():
    counter = Counter(emotion_history)
    total = sum(counter.values())
    return {emotion: (counter[emotion] / total * 100) if total > 0 else 0 for emotion in all_emotions}

# Stress Score Calculation
def calculate_stress_score(emotion_distribution):
    w1, w2, w3, w4, w5 = 1.0, 0.8, 0.7, 0.6, 0.5
    P_Angry = emotion_distribution.get('anger', 0)
    P_Fear = emotion_distribution.get('fear', 0)
    P_Sad = emotion_distribution.get('sad', 0)
    P_Disgust = emotion_distribution.get('disgust', 0)
    P_Happy = emotion_distribution.get('happy', 0)
    
    S = w1 * P_Angry + w2 * P_Fear + w3 * P_Sad + w4 * P_Disgust - w5 * P_Happy

    return max(S, 0)

# Tiredness Score Calculation
def calculate_tiredness_score(emotion_distribution):
    v1, v2, v3, v4 = 1.0, 0.8, 0.6, 0.5
    P_Sad = emotion_distribution.get('sad', 0)
    P_Neutral = emotion_distribution.get('neutral', 0)
    P_Happy = emotion_distribution.get('happy', 0)
    P_Surprise = emotion_distribution.get('surprise', 0)
    
    T = v1 * P_Sad + v2 * P_Neutral - v3 * P_Happy - v4 * P_Surprise

    return max(T, 0)

# Thread function to continuously detect faces on a downscaled frame
def process_frame():
    global global_frame, face_locations
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            frame_copy = global_frame.copy()
        small_frame = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        small_face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in small_face_locations]

# Start the face detection thread
detection_thread = threading.Thread(target=process_frame, daemon=True)
detection_thread.start()

# Set up the log file and write header if file is empty
log_file = 'emotion_history.csv'
with open(log_file, 'a') as f:
    if f.tell() == 0:
        f.write("timestamp,stress_score,tiredness_score\n")

# Record last logging time (initialize to current time)
last_log_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 == 0:
        with frame_lock:
            global_frame = frame.copy()

    current_face_locations = face_locations.copy()
    faces = []
    for (top, right, bottom, left) in current_face_locations:
        face_image = frame[top:bottom, left:right]
        try:
            face_resized = cv2.resize(face_image, (224, 224))
        except Exception:
            continue
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        faces.append(face_pil)
    
    emotions = []
    if faces:
        if choice == "4":
            for face_pil in faces:
                votes = []
                for proc, mod in zip(processors, models):
                    inputs = proc(images=face_pil, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = mod(**inputs)
                    pred = outputs.logits.argmax(-1).item()
                    votes.append(mod.config.id2label[pred])

                print(votes)
                for i in range(len(votes)):
                    votes[i] = label_map.get(votes[i].lower(), votes[i].lower())

                print(votes)
                majority_label = max(set(votes), key=votes.count)
                normalized = label_map.get(majority_label.lower(), majority_label.lower())
                emotions.append(normalized)
        
        else:
            inputs = processor(images=faces, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1).cpu().numpy()
            emotions = [model.config.id2label[pred] for pred in predictions]
            
            emotions = [label_map.get(e.lower(), e.lower()) for e in emotions]
            print(emotions)
        
        for (top, right, bottom, left), emotion in zip(current_face_locations, emotions):
            emotion_history.append(emotion)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    distribution = get_emotion_distribution()
    y_offset = 20
    cv2.putText(frame, "Mood", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for emotion, percentage in distribution.items():
        y_offset += 30
        text = f"{percentage:.0f}% {emotion}"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate Stress and Tiredness Scores
    stress_score = calculate_stress_score(distribution)
    tiredness_score = calculate_tiredness_score(distribution)

    if tiredness_score > TIREDNESS_ALERT_THRESHOLD:
    # draw a big warning on the frame
        cv2.putText(frame,
                    "You look tired! Take a break!",
                    (10, y_offset + 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (0, 0, 255), 2)
        # fire the alert sound in its own thread so it doesn't block your frame loop
        threading.Thread(target=sound_alert, daemon=True).start()
    
    if stress_score > STRESS_ALERT_THRESHOLD:
        # overlay a red warning
        cv2.putText(frame,
                    "High stress! Take a moment to relax!",
                    (10, y_offset + 130),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8, (0, 0, 255), 2)
        # play alert sound without blocking
        threading.Thread(target=sound_alert, daemon=True).start()

    
    cv2.putText(frame, f"Stress: {stress_score:.2f}", (10, y_offset + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Tiredness: {tiredness_score:.2f}", (10, y_offset + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Log the scores only every 5 seconds
    current_time_sec = time.time()
    if current_time_sec - last_log_time >= 5:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"{current_time},{stress_score:.2f},{tiredness_score:.2f}\n")
        last_log_time = current_time_sec
    
    cv2.imshow('Optimized Face Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
