import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import subprocess
import csv

# For inline image prediction and face detection
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import face_recognition

# Global model choices and state
MODEL_CHOICES = {
    '1': 'trpakov/vit-face-expression',
    '2': 'dima806/facial_emotions_image_detection',
    '3': 'motheecreator/vit-Facial-Expression-Recognition',
    '4': 'combined'
}
current_choice = '3'

# Launch external live recognition script
def start_live():
    cmd = [sys.executable, 'fast.py']
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        p.communicate(current_choice + '\n')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start live recognition:\n{e}")

# Load picture, detect faces, predict emotions, and display annotated image
def load_picture():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return
    model_name = MODEL_CHOICES[current_choice]
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if current_choice == '4':
            model_names = list(MODEL_CHOICES.values())[:3]
            processors = [AutoImageProcessor.from_pretrained(name) for name in model_names]
            models = [ViTForImageClassification.from_pretrained(name).to(device) for name in model_names]
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name).to(device)

        # Load image
        pil_image = Image.open(path).convert("RGB")
        image_array = face_recognition.load_image_file(path)
        # Detect faces
        locations = face_recognition.face_locations(image_array, model='hog')
        if not locations:
            messagebox.showinfo("No Face Detected", "Could not find any faces in the image.")
            return

        # Prepare annotation
        draw = ImageDraw.Draw(pil_image)
        
        font = ImageFont.load_default()

        # Normalize label similar to combined mode
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
            'neutral': 'neutral'
        }

        # Predict emotion for each detected face
        for (top, right, bottom, left) in locations:
            # Crop and resize
            face = pil_image.crop((left, top, right, bottom)).resize((224, 224))
            if current_choice == '4':
                votes = []
                for proc, mod in zip(processors, models):
                    inputs = proc(images=face, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = mod(**inputs)
                    pred = outputs.logits.argmax(-1).item()
                    votes.append(mod.config.id2label[pred])

                print(votes)
                for i in range(len(votes)):
                    votes[i] = label_map.get(votes[i].lower(), votes[i].lower())

                print(votes)
                majority = max(set(votes), key=votes.count)
                label = label_map.get(majority.lower(), majority.lower())
            else:
                inputs = processor(images=face, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = outputs.logits.argmax(-1).item()
                label = model.config.id2label[pred]
                label = label_map.get(label.lower(), label.lower())

            # Draw rectangle and label
            draw.rectangle(((left, top), (right, bottom)), outline="green", width=3)
            text_pos = (left, top - 25 if top - 25 > 0 else top + 5)
            draw.text(text_pos, label, fill="green", font=font)

        # Display annotated image
        win = tk.Toplevel(root)
        win.title(f"Predicted Emotions")
        # Resize for display
        disp = pil_image.copy()
        disp.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(disp)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

        # Also show a summary messagebox
        messagebox.showinfo("Prediction", "Emotions predicted and displayed on image window.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict emotion:\n{e}")

# View historical logs
def view_logs():
    win = tk.Toplevel(root)
    win.title("Emotion Logs")
    win.geometry('600x400')

    columns = ('timestamp', 'stress', 'tiredness')
    tree = ttk.Treeview(win, columns=columns, show='headings')
    for col, width in zip(columns, (200, 100, 120)):
        tree.heading(col, text=col.replace('_', ' ').title())
        tree.column(col, width=width, anchor='center')

    vsb = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
    tree.configure(yscroll=vsb.set)
    vsb.pack(side='right', fill='y')
    tree.pack(expand=True, fill='both')

    try:
        with open('emotion_history.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    tree.insert('', 'end', values=row[:3])
    except FileNotFoundError:
        messagebox.showinfo("Logs", "No logs found.")

# Model selection window
def select_model():
    win = tk.Toplevel(root)
    win.title("Select Model")
    var = tk.StringVar(value=current_choice)
    for key, name in MODEL_CHOICES.items():
        ttk.Radiobutton(win, text=f"{key} - {name}", variable=var, value=key).pack(anchor='w', padx=10, pady=2)
    def apply_choice():
        global current_choice
        current_choice = var.get()
        win.destroy()
    ttk.Button(win, text="Apply", command=apply_choice).pack(pady=10)

# Build main GUI
def build_gui():
    global root
    root = tk.Tk()
    root.title("Emotion Recognition")
    root.geometry('480x360')

    ttk.Label(root, text="Emotion Recognition", font=('Helvetica', 20, 'bold')).pack(pady=20)

    btns = [
        ("Start Live Recognition", start_live),
        ("Load Picture", load_picture),
        ("View Logs", view_logs),
        ("Select Model", select_model)
    ]
    frame = ttk.Frame(root)
    frame.pack(pady=10)
    for (txt, cmd) in btns:
        ttk.Button(frame, text=txt, width=30, command=cmd).pack(pady=5)

    bottom = ttk.Frame(root)
    bottom.pack(side='bottom', fill='x', pady=15)
    ttk.Button(bottom, text="Quit", command=root.quit).pack(side='left', padx=20)
    ttk.Button(bottom, text="GitHub", command=lambda: webbrowser.open('https://github.com/rubargunduz/EmotionRecognition')).pack(side='right', padx=20)

    root.mainloop()

if __name__ == '__main__':
    build_gui()
