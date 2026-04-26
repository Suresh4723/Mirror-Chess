import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- LOAD ----------------
data = torch.load("dataset_v4.pt")

boards = data["boards"]
extra = data["extra"]
labels = data["labels"]

# ---------------- DATASET ----------------
class ChessDataset(Dataset):
    def __len__(self):
        return len(labels)

    def __getitem__(self, i):
        return boards[i], extra[i], labels[i]

loader = DataLoader(
    ChessDataset(),
    batch_size=256,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ---------------- MODEL ----------------
class ChessCNN(nn.Module):
    def __init__(self, extra_size):
        super().__init__()

        self.conv1 = nn.Conv2d(14, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 8 * 8 + extra.shape[1], 512)
        self.fc2 = nn.Linear(512, 4096)

    def forward(self, board, extra):
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)

model = ChessCNN(extra.shape[1]).to(device)

# ---------------- TRAIN SETUP ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler("cuda")

# ---------------- TRAIN ----------------
EPOCHS = 13   # 🔥 IMPORTANT (based on your previous run)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")

    model.train()
    total_loss = 0

    for b, e, y in tqdm(loader):
        b = b.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(b, e)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    print(f"Loss: {total_loss/len(loader):.4f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "cnn_v4_final.pth")
print("✅ FINAL MODEL SAVED")