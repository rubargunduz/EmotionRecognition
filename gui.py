import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import subprocess
import csv

# For inline image prediction
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Global model choices and state
MODEL_CHOICES = {
    '1': 'trpakov/vit-face-expression',
    '2': 'dima806/facial_emotions_image_detection',
    '3': 'motheecreator/vit-Facial-Expression-Recognition'
}
current_choice = '3'

def start_live():
    cmd = [sys.executable, 'fast.py']  # no --model flag here
    try:
        # launch and immediately feed it the choice
        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             text=True)
        p.communicate(current_choice + '\n')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start live recognition:\n{e}")

def load_picture():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return
    # Inline prediction instead of external script
    model_name = MODEL_CHOICES[current_choice]
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = outputs.logits.argmax(-1).item()
        label = model.config.id2label[pred_id]
        messagebox.showinfo("Prediction", f"Predicted Emotion: {label}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict emotion:\n{e}")

def view_logs():
    win = tk.Toplevel(root)
    win.title("Emotion Logs")
    win.geometry('600x400')

    # Set up treeview
    columns = ('timestamp', 'stress', 'tiredness')
    tree = ttk.Treeview(win, columns=columns, show='headings')
    tree.heading('timestamp', text='Timestamp')
    tree.heading('stress', text='Stress Score')
    tree.heading('tiredness', text='Tiredness Score')
    tree.column('timestamp', width=200)
    tree.column('stress', width=100, anchor='center')
    tree.column('tiredness', width=120, anchor='center')

    # Add vertical scrollbar
    vsb = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
    tree.configure(yscroll=vsb.set)
    vsb.pack(side='right', fill='y')
    tree.pack(expand=True, fill='both')

    # Load CSV content
    try:
        with open('emotion_history.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 3:
                    tree.insert('', 'end', values=(row[0], row[1], row[2]))
    except FileNotFoundError:
        messagebox.showinfo("Logs", "No logs found.")


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

# Main GUI setup
def build_gui():
    global root
    root = tk.Tk()
    root.title("Emotion Recognition")
    root.geometry('480x360')

    # Greeting
    ttk.Label(root, text="Emotion Recognition", font=('Helvetica', 20, 'bold')).pack(pady=20)

    # Action buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)
    actions = [
        ("Start Live Recognition", start_live),
        ("Load Picture", load_picture),
        ("View Logs", view_logs),
        ("Select Model", select_model)
    ]
    for txt, cmd in actions:
        ttk.Button(btn_frame, text=txt, width=30, command=cmd).pack(pady=5)

    # Bottom controls
    bottom = ttk.Frame(root)
    bottom.pack(side='bottom', fill='x', pady=15)
    ttk.Button(bottom, text="Quit", command=root.quit).pack(side='left', padx=20)
    ttk.Button(
        bottom, text="GitHub", command=lambda: webbrowser.open('https://github.com/yourrepo')
    ).pack(side='right', padx=20)

    root.mainloop()

if __name__ == '__main__':
    build_gui()
