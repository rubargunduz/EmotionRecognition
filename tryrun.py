import os
import json
import torch
import random
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

emotions = ['happy', 'sad', 'fear', 'neutral', 'surprise', 'anger', 'disgust']
emotion_to_index = {e: i for i, e in enumerate(emotions)}

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

def one_hot(label):
    vec = [0] * 7
    vec[emotion_to_index[label]] = 1
    return vec

def load_face(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        return img
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None
    
MODELS = {
    '1': 'Thao2202/vit-Facial-Expression-Recognition',
    '2': 'Alpiyildo/vit-Facial-Expression-Recognition',
    '3': 'motheecreator/vit-Facial-Expression-Recognition',
}

processors = {}
models = {}

for key, name in MODELS.items():
    processor = AutoImageProcessor.from_pretrained(name)
    model = ViTForImageClassification.from_pretrained(name).to(device)
    processors[key] = processor
    models[key] = model


dataset_dir = "test_dataset"
samples = []

# Sort folders and shuffle files consistently
all_image_paths = []
for label in sorted(os.listdir(dataset_dir)):
    folder = os.path.join(dataset_dir, label)
    if not os.path.isdir(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        all_image_paths.append((os.path.join(folder, fname), label))

random.seed(42)
random.shuffle(all_image_paths)

# Take first 70% for training VotingNet
split_idx = int(len(all_image_paths) * 0.7)
training_samples = all_image_paths[:split_idx]

for path, label in training_samples:
    true_label = label_map.get(label.lower(), label.lower())
    face = load_face(path)
    if face is None:
        continue

    predictions = []
    for i, (processor, model) in enumerate(zip(processors.values(), models.values())):
        inputs = processor(images=face, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
        pred_label = model.config.id2label[pred].lower()
        predictions.append(label_map.get(pred_label, pred_label))

    samples.append({
        "pred1": predictions[0],
        "pred2": predictions[1],
        "pred3": predictions[2],
        "true_label": true_label
    })

# Save dataset
with open("votingnet_training_data.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"Saved {len(samples)} training samples for VotingNet.")



# Load dataset
with open("votingnet_training_data.json", "r") as f:
    data = json.load(f)

X = torch.tensor([one_hot(d["pred1"]) + one_hot(d["pred2"]) + one_hot(d["pred3"]) for d in data], dtype=torch.float32)
y = torch.tensor([emotion_to_index[d["true_label"]] for d in data], dtype=torch.long)

# Define the model
class VotingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

model = VotingNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(50):
    model.train()
    inputs = X.to(device)
    labels = y.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        acc = (outputs.argmax(1) == labels).float().mean().item() * 100
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.2f}%")


torch.save(model.state_dict(), "votingnet_model.pt")
print("Saved trained VotingNet model to 'votingnet_model.pt'")


import torch.nn as nn
import torch.nn.functional as F

# VotingNet and one_hot for voting ensemble
class VotingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def one_hot(label):
    vec = [0] * 7
    emotions = ['happy', 'sad', 'fear', 'neutral', 'surprise', 'anger', 'disgust']
    emotion_to_index = {e: i for i, e in enumerate(emotions)}
    vec[emotion_to_index[label]] = 1
    return vec
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


# Accuracy per emotion per model (manually copied from test results)
emotion_weights = {
    'happy':    [0.85, 1.0, 0.98],
    'sad':      [0.73, 1.0, 0.98],
    'fear':     [0.64, 0.93, 1.0],
    'neutral':  [0.71, 1.0, 0.99],
    'surprise': [0.84, 1.0, 0.97],
    'anger':    [0.73, 1.0, 0.96],
    'disgust':  [0.71, 0.99, 1.0]
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

    return correct / total * 100 if total > 0 else 0


def evaluate_combined(models, processors, dataset_dir):
    total, correct = 0, 0
    per_label_stats = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_voting = VotingNet().to(device)
    model_voting.load_state_dict(torch.load("votingnet_model.pt", map_location=device))
    model_voting.eval()

    # Prepare test data: last 30% of all images in dataset_dir
    all_image_paths = []
    for label in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            all_image_paths.append((os.path.join(folder, fname), label))

    split_idx = int(len(all_image_paths) * 0.7)
    test_samples = all_image_paths[split_idx:]

    for path, label in test_samples:
        true_label = label_map.get(label.lower(), label.lower())
        face = load_face(path)
        if face is None:
            continue

        predictions = []
        for processor, model in zip(processors, models):
            inputs = processor(images=face, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            pred_label = model.config.id2label[pred].lower()
            predictions.append(label_map.get(pred_label, pred_label))

        # Compose input for VotingNet: one-hot for each model prediction, concatenated
        input_tensor = torch.tensor([one_hot(predictions[0]) + one_hot(predictions[1]) + one_hot(predictions[2])], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model_voting(input_tensor)
        majority = ['happy', 'sad', 'fear', 'neutral', 'surprise', 'anger', 'disgust'][output.argmax().item()]

        total += 1
        if true_label not in per_label_stats:
            per_label_stats[true_label] = [0, 0]
        per_label_stats[true_label][1] += 1
        if majority == true_label:
            correct += 1
            per_label_stats[true_label][0] += 1

    print(f"Evaluated {total} images using VotingNet.")
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
