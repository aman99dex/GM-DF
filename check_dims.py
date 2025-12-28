
from PIL import Image
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python check_dims.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
try:
    img = Image.open(path)
    print(f"Dimensions: {img.size}")
    if img.size[0] == img.size[1]:
        print("Aspect Ratio: 1:1 (Likely Cropped)")
    else:
        print(f"Aspect Ratio: {img.size[0]/img.size[1]:.2f} (Likely Raw Frame)")
except Exception as e:
    print(f"Error: {e}")
