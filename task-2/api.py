from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            
            # in channels, out channels, kernel size
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)          # -> (batch, 128, 1, 1)
        x = torch.flatten(x, 1)   # -> (batch, 128)
        x = self.classifier(x)
        return x

app = FastAPI()

# Load model
model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Optional label names
labels = ["cat", "dog", "car"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return {
        "filename": file.filename,
        "predicted_class_index": predicted_class,
        "predicted_label": labels[predicted_class]
    }