import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
MODEL_NAMES = {
    '1': 'dima806/facial_emotions_image_detection',
    '2': 'trpakov/vit-face-expression',
    '3': 'motheecreator/vit-Facial-Expression-Recognition',
}

DATASET_DIR = "/kaggle/input/fer2013/train"

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
}
emotions = ['happy', 'sad', 'fear', 'neutral', 'surprise', 'angry', 'disgust']
emotion_to_index = {e: i for i, e in enumerate(emotions)}

class FERDataset(Dataset):
    def __init__(self, image_label_pairs, processor):
        self.data = image_label_pairs
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label_id = emotion_to_index[label_map.get(label.lower(), label.lower())]
        inputs["labels"] = torch.tensor(label_id)
        return inputs

def load_data(dataset_dir, split=0.1):
    image_label_pairs = []
    for label in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            image_label_pairs.append((os.path.join(folder, fname), label))

    train_data, val_data = train_test_split(image_label_pairs, test_size=split, stratify=[x[1] for x in image_label_pairs])
    return train_data, val_data

def collate_fn(batch):
    input_ids = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'pixel_values': torch.stack(input_ids),
        'labels': torch.tensor(labels)
    }

def fine_tune(model_id, output_dir):
    print(f"Fine-tuning model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id, num_labels=len(emotions)).to(device)

    train_data, val_data = load_data(DATASET_DIR)

    train_dataset = FERDataset(train_data, processor)
    val_dataset = FERDataset(val_data, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=5e-5,
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    for key, model_name in MODEL_NAMES.items():
        fine_tune(model_name, f"./fine_tuned_model_{key}")