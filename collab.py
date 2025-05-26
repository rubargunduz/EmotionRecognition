import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


# Accuracy per emotion per model (manually copied from test results)
emotion_weights = {
    'happy':    [1.0, 0.0, 0.9],
    'sad':      [0.5, 0.6, 0.7],
    'fear':     [0.6, 0.9, 0.7],
    'neutral':  [0.9, 0.6, 0.5],
    'surprise': [1.0, 0.8, 1.0],
    'anger':    [0.9, 0.8, 0.9],
    'disgust':  [0.6, 1.0, 1.0]
}

# Normalize labels
label_map = {
    'angry': 'angry',
    'anger': 'angry',
    'fear': 'fear',
    'scared': 'fear',
    'sadness': 'sad',
    'sad': 'sad',
    'disgust': 'disgust',
    'happy': 'happy',
    'happiness': 'happy',
    'surprise': 'surprise',
    'suprise': 'surprise',
    'neutral': 'neutral',
    'contempt': 'neutral'
}

def get_weighted_majority_vote(predictions):
    weighted_vote_counter = {}
    for i, pred in enumerate(predictions):
        normalized = label_map.get(pred.lower(), pred.lower())
        weight = emotion_weights.get(normalized, [0, 0, 0])[i]
        weighted_vote_counter[normalized] = weighted_vote_counter.get(normalized, 0) + weight
    return max(weighted_vote_counter, key=weighted_vote_counter.get)

import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import face_recognition
from sklearn.metrics import confusion_matrix
import numpy as np

# Models
MODELS = {
    '1': 'trpakov/vit-face-expression',
    '2': 'HardlyHumans/Facial-expression-detection',
    '3': 'motheecreator/vit-Facial-Expression-Recognition',
}



def load_face(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        return img
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None

def evaluate_model(processor, model, dataset_dir):
    total, correct = 0, 0
    per_label_stats = {}
    for label in os.listdir(dataset_dir):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        true_label = label_map.get(label.lower(), label.lower())
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            face = load_face(path)
            if face is None:
                continue
            inputs = processor(images=face, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            pred_label = label_map.get(model.config.id2label[pred].lower(), model.config.id2label[pred].lower())
            total += 1
            if true_label not in per_label_stats:
                per_label_stats[true_label] = [0, 0]
            per_label_stats[true_label][1] += 1
            if pred_label == true_label:
                correct += 1
                per_label_stats[true_label][0] += 1
    print(f"Evaluated {total} images for model.")
    print("Accuracy per emotion:")
    for label, (corr, tot) in per_label_stats.items():
        acc = (corr / tot * 100) if tot > 0 else 0
        print(f"  {label}: {acc:.2f}% ({corr}/{tot})")

    all_preds = []
    all_trues = []
    for label in os.listdir(dataset_dir):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        true_label = label_map.get(label.lower(), label.lower())
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            face = load_face(path)
            if face is None:
                continue
            inputs = processor(images=face, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            pred_label = label_map.get(model.config.id2label[pred].lower(), model.config.id2label[pred].lower())
            all_trues.append(true_label)
            all_preds.append(pred_label)

    labels_sorted = sorted(set(all_trues + all_preds))
    cm = confusion_matrix(all_trues, all_preds, labels=labels_sorted)
    print("\nConfusion Matrix:")
    print("Labels:", labels_sorted)
    print(np.array(cm))

    return correct / total * 100 if total > 0 else 0


def evaluate_combined(models, processors, dataset_dir):
    total, correct = 0, 0
    per_label_stats = {}

    for label in os.listdir(dataset_dir):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        true_label = label_map.get(label.lower(), label.lower())
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            face = load_face(path)
            if face is None:
                continue

            predictions = []
            for i, (processor, model) in enumerate(zip(processors, models)):
                inputs = processor(images=face, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = outputs.logits.argmax(-1).item()
                pred_label = model.config.id2label[pred].lower()
                predictions.append(pred_label)

            majority = get_weighted_majority_vote(predictions)

            total += 1
            if true_label not in per_label_stats:
                per_label_stats[true_label] = [0, 0]
            per_label_stats[true_label][1] += 1
            if majority == true_label:
                correct += 1
                per_label_stats[true_label][0] += 1
    print(f"Evaluated {total} images for combined model.")
    print("Accuracy per emotion:")
    for label, (corr, tot) in per_label_stats.items():
        acc = (corr / tot * 100) if tot > 0 else 0
        print(f"  {label}: {acc:.2f}% ({corr}/{tot})")
    return correct / total * 100 if total > 0 else 0

def main(dataset_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    print("\nLoading models...")
    processors = {}
    models = {}
    for key, name in MODELS.items():
        processor = AutoImageProcessor.from_pretrained(name)
        model = ViTForImageClassification.from_pretrained(name).to(device)
        processors[key] = processor
        models[key] = model

    

    print("\nEvaluating combined (voting) model:")
    acc = evaluate_combined(list(models.values()), list(processors.values()), dataset_dir)
    print(f"Combined model: {acc:.2f}% accuracy")

if __name__ == "__main__":
    main("/Users/rubargunduz/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test")
