from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load the model and processor
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load and preprocess an image
image = Image.open("testImage.jpg")  # Replace with the path to your image
inputs = processor(images=image, return_tensors="pt")

# Make a prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class
predicted_label = outputs.logits.argmax(-1).item()
print("Predicted Emotion:", model.config.id2label[predicted_label])
