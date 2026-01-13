
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

# ----- INFERENCE IMAGES (Square and Pentagon) -----
# Create separate inference directory
INFERENCE_DIR = "cnn_toy_data_inference"
INFERENCE_IMG_DIR = os.path.join(INFERENCE_DIR, "images")
os.makedirs(INFERENCE_IMG_DIR, exist_ok=True)

inference_images = []

# ----- SQUARE (new shape, unlabeled) -----
square = np.zeros((8, 8))
square[2:6, 2:6] = 255  # 4x4 square in center
img_square = Image.fromarray(square.astype(np.uint8), mode="L")
img_square.save(os.path.join(INFERENCE_IMG_DIR, "square_1.png"))
pd.DataFrame(square).to_csv(
    os.path.join(INFERENCE_DIR, "square_1.csv"),
    index=False,
    header=False
)
inference_images.append(("square_1", "new_shape"))

# ----- PENTAGON (new shape, unlabeled) -----
pentagon = np.zeros((8, 8))
# Rough pentagon centered in 8x8 grid
pentagon[1, 3:5] = 255       # Top point (row 1)
pentagon[2, 2:6] = 255       # Upper sides
pentagon[3, 2:6] = 255       # Middle
pentagon[4, 1:7] = 255       # Wider middle
pentagon[5, 1:7] = 255       # Lower middle
pentagon[6, 2:6] = 255       # Lower sides
img_pentagon = Image.fromarray(pentagon.astype(np.uint8), mode="L")
img_pentagon.save(os.path.join(INFERENCE_IMG_DIR, "pentagon_1.png"))
pd.DataFrame(pentagon).to_csv(
    os.path.join(INFERENCE_DIR, "pentagon_1.csv"),
    index=False,
    header=False
)
inference_images.append(("pentagon_1", "new_shape"))

# Save inference metadata
inference_df = pd.DataFrame({
    "image": [img[0] for img in inference_images],
    "description": [img[1] for img in inference_images]
})
inference_df.to_csv(os.path.join(INFERENCE_DIR, "inference_images.csv"), index=False)

print("Toy image dataset created successfully.")
print(f"Base directory: {BASE_DIR}")
print(f"Inference directory: {INFERENCE_DIR}")
print("  - square_1.png (new shape)")
print("  - pentagon_1.png (new shape)")
