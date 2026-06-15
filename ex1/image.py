# Import required libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 1. Load an image
img = Image.open('abra.png')  
# Convert image to numpy array
img_array = np.array(img)
# 2. Display the image
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()
# dimensions
height, width = img_array.shape[0], img_array.shape[1]
print("Height:", height)
print("Width:", width)
# 4. grayscale or RGB
if len(img_array.shape) == 2:
    print("Image is Grayscale")
    # Convert grayscale to RGB
    rgb_img = img.convert("RGB")
    plt.imshow(rgb_img)
    plt.title("Converted to RGB")
    plt.axis('off')
    plt.show()
elif len(img_array.shape) == 3:
    print("Image is RGB")
    # Convert RGB to Grayscale
    gray_img = img.convert("L")
    plt.imshow(gray_img, cmap='gray')
    plt.title("Converted to Grayscale")
    plt.axis('off')
    plt.show()
# 5. Calculate total number of pixels
total_pixels = height * width
print("Total number of pixels:", total_pixels)