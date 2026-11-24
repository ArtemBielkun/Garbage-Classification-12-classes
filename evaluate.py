import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "garbage_classification"  
WEIGHTS_PATH = "model_weights_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes


model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


y_true = []
y_pred = []

print("Обчислення метрик... Це займе 1-2 хвилини.")
with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


print("\n=== РЕЗУЛЬТАТИ ДЛЯ ТАБЛИЦІ 3.3 ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=2))


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Передбачений клас")
plt.ylabel("Справжній клас")
plt.title("Матриця помилок (Confusion Matrix) – ResNet50")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Готово! Картинку збережено як confusion_matrix.png")