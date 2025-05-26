def evaluate_single_model(model_name, dataset_dir):
    from transformers import AutoImageProcessor, ViTForImageClassification
    import torch
    from PIL import Image
    import os
    from sklearn.metrics import confusion_matrix
    import numpy as np
    from utils import label_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)

    def load_face(image_path):
        try:
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            return img
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            return None

    correct, total = 0, 0
    per_label_stats = {}
    all_preds, all_trues = [], []

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

            inputs = processor(images=face, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            pred_label = label_map.get(model.config.id2label[pred].lower(), model.config.id2label[pred].lower())

            all_trues.append(true_label)
            all_preds.append(pred_label)

            total += 1
            if true_label not in per_label_stats:
                per_label_stats[true_label] = [0, 0]
            per_label_stats[true_label][1] += 1
            if pred_label == true_label:
                correct += 1
                per_label_stats[true_label][0] += 1

    print(f"\nEvaluated {total} images with model: {model_name}")
    print("Accuracy per emotion:")
    for label, (corr, tot) in per_label_stats.items():
        acc = (corr / tot * 100) if tot > 0 else 0
        print(f"  {label}: {acc:.2f}% ({corr}/{tot})")

    labels_sorted = sorted(set(all_trues + all_preds))
    cm = confusion_matrix(all_trues, all_preds, labels=labels_sorted)
    print("\nConfusion Matrix:")
    print("Labels:", labels_sorted)
    print(np.array(cm))

    return correct / total * 100 if total > 0 else 0

accuracy = evaluate_single_model("Rajaram1996/FacialEmoRecog", "/kaggle/input/fer2013/test")
print(f"\nFinal Accuracy: {accuracy:.2f}%")