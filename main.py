import cv2
import face_recognition
import threading

# Initialize webcam
video_capture = cv2.VideoCapture(0)
face_locations = []
frame = None
frame_count = 0

# Function to process frames in a separate thread
def process_frame():
    global frame, face_locations
    while True:
        if frame is not None:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce size by 50%
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]

# Start the processing thread
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 == 0:  # Process every 3rd frame
        frame = frame.copy()
    
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Optimized Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
