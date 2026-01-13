
# CNN Demo (Story Mode)
# This script generates ALL CNN artifacts + a STORY.md walkthrough
# Run AFTER generate_toy_images.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from PIL import Image

BASE_DIR = "cnn_toy_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABELS = os.path.join(BASE_DIR, "labels.csv")
OUT_DIR = "cnn_demo_story_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

labels_df = pd.read_csv(LABELS)

class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2 * 3 * 3, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ToyCNN()
criterion = nn.CrossEntropyLoss()

# Save kernels and weights
for f in range(model.conv1.weight.shape[0]):
    pd.DataFrame(model.conv1.weight[f,0].detach().numpy()).to_csv(
        f"{OUT_DIR}/GLOBAL_conv1_kernel_filter{f}.csv", index=False
    )

pd.DataFrame(model.fc.weight.detach().numpy()).to_csv(
    f"{OUT_DIR}/GLOBAL_fc_weights.csv", index=False
)

# Process images
for _, row in labels_df.iterrows():
    name = row["image"]
    label = torch.tensor([row["label"]])

    img = Image.open(os.path.join(IMG_DIR, f"{name}.png")).convert("L")
    arr = np.array(img)

    pd.DataFrame(arr).to_csv(f"{OUT_DIR}/{name}_01_pixels.csv", index=False)

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    conv = model.conv1(x)
    relu = F.relu(conv)
    pool = model.pool(relu)
    flat = pool.view(1, -1)
    logits = model.fc(flat)
    probs = F.softmax(logits, dim=1)
    loss = criterion(logits, label)

    for i in range(2):
        pd.DataFrame(conv[0,i].detach().numpy()).to_csv(
            f"{OUT_DIR}/{name}_02_conv_f{i}.csv", index=False
        )
        pd.DataFrame(relu[0,i].detach().numpy()).to_csv(
            f"{OUT_DIR}/{name}_03_relu_f{i}.csv", index=False
        )
        pd.DataFrame(pool[0,i].detach().numpy()).to_csv(
            f"{OUT_DIR}/{name}_04_pool_f{i}.csv", index=False
        )

    pd.DataFrame(flat.detach().numpy()).to_csv(f"{OUT_DIR}/{name}_05_flat.csv", index=False)
    pd.DataFrame(logits.detach().numpy()).to_csv(f"{OUT_DIR}/{name}_06_logits.csv", index=False)
    pd.DataFrame(probs.detach().numpy()).to_csv(f"{OUT_DIR}/{name}_07_softmax.csv", index=False)

    with open(f"{OUT_DIR}/{name}_08_loss.txt", "w") as f:
        f.write(str(loss.item()))

# Generate STORY.md
with open(f"{OUT_DIR}/STORY.md", "w") as f:
    f.write("""# CNN Toy Example – Guided Walkthrough

## Step 1: Image as Data
Open *_01_pixels.csv

## Step 2: Kernels (Learned Weights)
Open GLOBAL_conv1_kernel_filter*.csv

## Step 3: Feature Maps
Compare *_02_conv_f0.csv across images

## Step 4: ReLU
Open *_03_relu_f*.csv

## Step 5: Pooling
Open *_04_pool_f*.csv

## Step 6: Flatten
Open *_05_flat.csv

## Step 7: Dense + Output
Open *_06_logits.csv and *_07_softmax.csv

## Step 8: Loss
Open *_08_loss.txt

CNN = feature learning + standard NN training
""")

print("Story-mode CNN demo ready. Open STORY.md first.")
