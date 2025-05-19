import os
# Supports common formats: JPEG, PNG, WEBP, AVIF (if Pillow compiled with libavif/libwebp)
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import face_recognition

# Models
MODELS = {
    '1': 'trpakov/vit-face-expression',
    '2': 'HardlyHumans/Facial-expression-detection',
    '3': 'motheecreator/vit-Facial-Expression-Recognition',
}

# Normalize labels
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
    'suprise': 'surprise',
    'neutral': 'neutral',
    'contempt': 'neutral'
}

def load_face(image_path):
    try:
        img = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(img, model='hog')
        if not locations:
            print(f"No face detected in: {image_path}")
            return None
        top, right, bottom, left = locations[0]
        face = Image.fromarray(img[top:bottom, left:right]).resize((224, 224))
        return face
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
        folder = os.path.join(dataset_dir, label)
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
    return correct / total * 100 if total > 0 else 0

def evaluate_combined(models, processors, dataset_dir):
    total, correct = 0, 0
    per_label_stats = {}
    for label in os.listdir(dataset_dir):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        true_label = label_map.get(label.lower(), label.lower())
        folder = os.path.join(dataset_dir, label)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            face = load_face(path)
            if face is None:
                continue
            votes = []
            for processor, model in zip(processors, models):
                inputs = processor(images=face, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = outputs.logits.argmax(-1).item()
                pred_label = model.config.id2label[pred]
                votes.append(label_map.get(pred_label.lower(), pred_label.lower()))
            majority = max(set(votes), key=votes.count)
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

    print("\nEvaluating individual models:")
    for key, model_id in MODELS.items():
        acc = evaluate_model(processors[key], models[key], dataset_dir)
        print(f"Model {key} ({model_id}): {acc:.2f}% accuracy")

    print("\nEvaluating combined (voting) model:")
    acc = evaluate_combined(list(models.values()), list(processors.values()), dataset_dir)
    print(f"Combined model: {acc:.2f}% accuracy")

if __name__ == "__main__":
    main("test_dataset")