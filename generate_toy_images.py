
"""
Stage 1: Generate 8x8 grayscale toy images for CNN understanding
Creates:
- 3 rectangle images (label = 0)
- 2 triangle images (label = 1)

Outputs:
- PNG images
- CSV pixel matrices
- labels.csv

Author: CNN Toy Demo
"""

from PIL import Image
import numpy as np
import pandas as pd
import os

# Output directories
BASE_DIR = "cnn_toy_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_DIR = os.path.join(BASE_DIR, "csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

images = []
labels = []

def save_image_and_csv(matrix, name, label):
    img = Image.fromarray(matrix.astype(np.uint8), mode="L")
    img.save(os.path.join(IMG_DIR, f"{name}.png"))

    pd.DataFrame(matrix).to_csv(
        os.path.join(CSV_DIR, f"{name}.csv"),
        index=False,
        header=False
    )

    images.append(name)
    labels.append(label)

# ----- RECTANGLES (label = 0) -----
for i in range(3):
    rect = np.zeros((8, 8))
    rect[2:6, 1:7] = 255
    save_image_and_csv(rect, f"rectangle_{i+1}", label=0)

# ----- TRIANGLES (label = 1) -----
tri1 = np.zeros((8, 8))
for i in range(2, 6):
    tri1[i, 4-(i-2):4+(i-2)+1] = 255
save_image_and_csv(tri1, "triangle_1", label=1)

tri2 = np.zeros((8, 8))
for i in range(1, 5):
    tri2[i, 4-(i-1):4+(i-1)+1] = 255
save_image_and_csv(tri2, "triangle_2", label=1)

# Save labels
label_df = pd.DataFrame({
    "image": images,
    "label": labels
})
label_df.to_csv(os.path.join(BASE_DIR, "labels.csv"), index=False)

print("Toy image dataset created successfully.")
print(f"Base directory: {BASE_DIR}")
