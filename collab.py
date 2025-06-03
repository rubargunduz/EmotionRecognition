
import json
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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