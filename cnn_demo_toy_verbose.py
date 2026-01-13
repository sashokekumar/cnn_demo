
"""
Stage 2 (Verbose): CNN Demo on Toy 8x8 Images

This script logs:
- Raw input pixels
- Convolution KERNEL WEIGHTS (filters)
- Convolution biases
- Feature maps (Conv / ReLU / Pool)
- Shapes at each stage
- Flattened vectors
- Dense layer weights & biases
- Logits, Softmax, Loss

All outputs are written to CSV/TXT for inspection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = "cnn_toy_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_OUT = "cnn_demo_outputs_verbose"

os.makedirs(CSV_OUT, exist_ok=True)

# -----------------------------
# Load labels
# -----------------------------
labels_df = pd.read_csv(os.path.join(BASE_DIR, "labels.csv"))

# -----------------------------
# CNN Model Definition
# -----------------------------
class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2 * 3 * 3, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ToyCNN()

criterion = nn.CrossEntropyLoss()

# -----------------------------
# GLOBAL: Save model parameters
# -----------------------------

# Convolution kernels (filters)
conv_weights = model.conv1.weight.detach().numpy()  # (2,1,3,3)
for f in range(conv_weights.shape[0]):
    pd.DataFrame(conv_weights[f, 0]).to_csv(
        f"{CSV_OUT}/GLOBAL_conv1_filter{f}_kernel.csv", index=False
    )

# Convolution biases
pd.DataFrame(model.conv1.bias.detach().numpy()).to_csv(
    f"{CSV_OUT}/GLOBAL_conv1_bias.csv", index=False
)

# Dense layer weights and bias
pd.DataFrame(model.fc.weight.detach().numpy()).to_csv(
    f"{CSV_OUT}/GLOBAL_fc_weights.csv", index=False
)
pd.DataFrame(model.fc.bias.detach().numpy()).to_csv(
    f"{CSV_OUT}/GLOBAL_fc_bias.csv", index=False
)

# -----------------------------
# Metadata / terminology log
# -----------------------------
with open(f"{CSV_OUT}/README_TERMINOLOGY.txt", "w") as f:
    f.write("""
CNN TERMINOLOGY MAPPING (THIS RUN)

INPUT:
- *_01_input_pixels.csv
  Raw 8x8 image matrix (grayscale)

CONVOLUTION:
- GLOBAL_conv1_filter*_kernel.csv
  3x3 convolution kernels (learnable weights)
- GLOBAL_conv1_bias.csv
  Bias per filter
- *_02_conv_filter*.csv
  Feature maps after convolution (before ReLU)

ACTIVATION:
- *_03_relu_filter*.csv
  Feature maps after ReLU

POOLING:
- *_04_pool_filter*.csv
  Feature maps after 2x2 max pooling

FLATTEN:
- *_05_flatten.csv
  1D feature vector (CNN -> tabular transition)

DENSE:
- GLOBAL_fc_weights.csv
  Fully connected layer weights
- GLOBAL_fc_bias.csv
  Fully connected layer bias
- *_06_logits.csv
  Raw class scores

OUTPUT:
- *_07_softmax.csv
  Class probabilities
- *_08_loss.txt
  Cross-entropy loss (scalar)

SHAPE FLOW:
Input        : (1, 1, 8, 8)
Conv output  : (1, 2, 6, 6)
Pool output  : (1, 2, 3, 3)
Flatten      : (1, 18)
Dense output : (1, 2)
""")

# -----------------------------
# Process each image
# -----------------------------
for idx, row in labels_df.iterrows():
    img_name = row["image"]
    label = torch.tensor([row["label"]])

    img_path = os.path.join(IMG_DIR, f"{img_name}.png")
    img = Image.open(img_path).convert("L")
    img_arr = np.array(img)

    pd.DataFrame(img_arr).to_csv(
        f"{CSV_OUT}/{img_name}_01_input_pixels.csv", index=False
    )

    x = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    conv_out = model.conv1(x)
    for f in range(conv_out.shape[1]):
        pd.DataFrame(conv_out[0, f].detach().numpy()).to_csv(
            f"{CSV_OUT}/{img_name}_02_conv_filter{f}.csv", index=False
        )

    relu_out = F.relu(conv_out)
    for f in range(relu_out.shape[1]):
        pd.DataFrame(relu_out[0, f].detach().numpy()).to_csv(
            f"{CSV_OUT}/{img_name}_03_relu_filter{f}.csv", index=False
        )

    pool_out = model.pool(relu_out)
    for f in range(pool_out.shape[1]):
        pd.DataFrame(pool_out[0, f].detach().numpy()).to_csv(
            f"{CSV_OUT}/{img_name}_04_pool_filter{f}.csv", index=False
        )

    flat = pool_out.view(1, -1)
    pd.DataFrame(flat.detach().numpy()).to_csv(
        f"{CSV_OUT}/{img_name}_05_flatten.csv", index=False
    )

    logits = model.fc(flat)
    pd.DataFrame(logits.detach().numpy()).to_csv(
        f"{CSV_OUT}/{img_name}_06_logits.csv", index=False
    )

    probs = F.softmax(logits, dim=1)
    pd.DataFrame(probs.detach().numpy()).to_csv(
        f"{CSV_OUT}/{img_name}_07_softmax.csv", index=False
    )

    loss = criterion(logits, label)
    with open(f"{CSV_OUT}/{img_name}_08_loss.txt", "w") as f:
        f.write(f"Loss: {loss.item()}")

print("Verbose CNN demo completed. Outputs written to:", CSV_OUT)
