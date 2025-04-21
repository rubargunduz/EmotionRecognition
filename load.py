from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Allow the user to choose a model at runtime
print("Select an emotion detection model:")
print("1 - trpakov/vit-face-expression")
print("2 - dima806/facial_emotions_image_detection")
print("3 - motheecreator/vit-Facial-Expression-Recognition")
choice = input("Enter model number: ").strip()

# Load the chosen model and processor
if choice == "2":
    model_name = "dima806/facial_emotions_image_detection"
elif choice == "3":
    model_name = "motheecreator/vit-Facial-Expression-Recognition"
else:
    model_name = "trpakov/vit-face-expression"  # Default model

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
