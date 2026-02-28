# ============================================================
# TERRITORIAL.IO - VISION MODEL TRAINING
# ============================================================
# HOW TO USE:
#   1. Go to https://kaggle.com/code → New Notebook
#   2. Delete all existing cells
#   3. Create ONE code cell and paste this ENTIRE script into it
#   4. On the right sidebar: Session options → Accelerator → GPU T4 x2
#   5. Click Run All (▶▶)
#   6. When done, go to the Output tab on the right
#   7. Download: vision_model.pth  AND  label_encoder.pkl
#   8. Put both files in your territorial_bot/models/ folder
# ============================================================

# ── Step 1: Install dependencies ──────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "torch", "torchvision", "opencv-python-headless",
                "Pillow", "numpy", "matplotlib", "scikit-learn"], check=True)

# ── Step 2: Imports ───────────────────────────────────────────
import os, random, pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── Step 3: Color palette (matches the real game) ─────────────
TERRITORY_COLORS = {
    'own': [
        (0, 100, 255), (0, 150, 255), (30, 120, 220), (0, 80, 200),
    ],
    'enemy': [
        (255, 50, 50), (220, 30, 30), (255, 80, 80), (200, 0, 0),
        (255, 140, 0), (180, 0, 255), (50, 200, 50),
        (255, 220, 0), (0, 220, 220), (255, 100, 180),
    ],
    'neutral': [
        (180, 180, 180), (160, 160, 160), (200, 200, 200),
        (140, 140, 140), (170, 170, 170),
    ],
    'border': [
        (10, 10, 10), (20, 20, 20), (5, 5, 5), (30, 30, 30),
    ],
}
CLASSES = ['own', 'enemy', 'neutral', 'border']
PATCH_SIZE = 64
SAMPLES_PER_CLASS = 2500

# ── Step 4: Generate synthetic training data ──────────────────
def generate_patch(label, size=64):
    colors = TERRITORY_COLORS[label]
    base = random.choice(colors)
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    if label == 'border':
        patch[:] = base
        noise = np.random.randint(0, 15, (size, size, 3), dtype=np.uint8)
        patch = np.clip(patch.astype(int) + noise - 7, 0, 255).astype(np.uint8)
        if random.random() > 0.5:
            cv2.line(patch, (0, size//2), (size, size//2), (50,50,50), 2)
        if random.random() > 0.5:
            cv2.line(patch, (size//2, 0), (size//2, size), (50,50,50), 2)

    elif label == 'neutral':
        patch[:] = base
        noise = np.random.randint(-20, 20, (size, size, 3))
        patch = np.clip(patch.astype(int) + noise, 0, 255).astype(np.uint8)
        if random.random() > 0.7:
            for i in range(0, size, 8):
                patch[i, :] = np.clip(patch[i, :].astype(int) - 10, 0, 255)
                patch[:, i] = np.clip(patch[:, i].astype(int) - 10, 0, 255)

    else:  # own or enemy
        patch[:] = base
        gradient = np.linspace(0, 30, size).astype(int)
        for i in range(size):
            patch[i] = np.clip(patch[i].astype(int) + gradient[i] - 15, 0, 255).astype(np.uint8)
        noise = np.random.randint(-15, 15, (size, size, 3))
        patch = np.clip(patch.astype(int) + noise, 0, 255).astype(np.uint8)
        if random.random() > 0.6:
            num = str(random.randint(1, 999))
            cv2.putText(patch, num, (random.randint(5,20), random.randint(20,50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        if random.random() > 0.5:
            ew = random.randint(1, 3)
            patch[:ew, :] = np.clip(patch[:ew, :].astype(int) - 40, 0, 255).astype(np.uint8)
    return patch

print("Generating synthetic training data...")
images, labels = [], []
for label in CLASSES:
    print(f"  Generating {SAMPLES_PER_CLASS} samples for: {label}")
    for _ in range(SAMPLES_PER_CLASS):
        images.append(generate_patch(label, PATCH_SIZE))
        labels.append(label)
images = np.array(images)
labels = np.array(labels)
print(f"✅ Dataset: {images.shape}")

# ── Step 5: Encode labels ─────────────────────────────────────
le = LabelEncoder()
le.fit(CLASSES)
label_ids = le.transform(labels)
with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"✅ Label encoder saved. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Step 6: Dataset class ─────────────────────────────────────
class TerritoryDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        if self.augment:
            if random.random() > 0.5: img = cv2.flip(img, 1)
            if random.random() > 0.5: img = cv2.flip(img, 0)
            k = random.randint(0, 3)
            if k > 0: img = np.rot90(img, k).copy()
            if random.random() > 0.5:
                img = np.clip(img.astype(float) * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
            if random.random() > 0.5:
                noise = np.random.normal(0, 10, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    images, label_ids, test_size=0.2, random_state=42, stratify=label_ids
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), norm])
val_tf   = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), norm])

train_loader = DataLoader(TerritoryDataset(X_train, y_train, train_tf, augment=True),
                          batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(TerritoryDataset(X_val, y_val, val_tf, augment=False),
                          batch_size=64, shuffle=False, num_workers=2)
print(f"✅ DataLoaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ── Step 7: CNN model (must match vision_system.py exactly) ───
class TerritoryClassifierCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = TerritoryClassifierCNN(num_classes=len(CLASSES)).to(device)
print(f"✅ Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 8: Training ──────────────────────────────────────────
EPOCHS = 30
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"\n🚀 Training for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, pred = outputs.max(1)
        t_total += targets.size(0)
        t_correct += pred.eq(targets).sum().item()

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            v_loss += loss.item()
            _, pred = outputs.max(1)
            v_total += targets.size(0)
            v_correct += pred.eq(targets).sum().item()

    train_acc = 100.0 * t_correct / t_total
    val_acc   = 100.0 * v_correct / v_total
    history['train_loss'].append(t_loss / len(train_loader))
    history['train_acc'].append(train_acc)
    history['val_loss'].append(v_loss / len(val_loader))
    history['val_acc'].append(val_acc)

    scheduler.step()

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/vision_model.pth')

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {t_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {v_loss/len(val_loader):.4f} Acc: {val_acc:.1f}% | "
              f"Best: {best_val_acc:.1f}%")

print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.1f}%")

# ── Step 9: Plot training curves ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_title('Loss'); ax1.legend(); ax1.set_xlabel('Epoch')
ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_title('Accuracy (%)'); ax2.legend(); ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=100)
plt.show()

# ── Step 10: Final summary ────────────────────────────────────
print("\n" + "="*50)
print("✅ FILES READY TO DOWNLOAD (check Output tab):")
print(f"   📁 {OUTPUT_DIR}/vision_model.pth    ← PUT IN territorial_bot/models/")
print(f"   📁 {OUTPUT_DIR}/label_encoder.pkl   ← PUT IN territorial_bot/models/")
print(f"   📁 {OUTPUT_DIR}/training_curves.png ← Training progress chart")
print("="*50)
print(f"Final best val accuracy: {best_val_acc:.1f}%")
