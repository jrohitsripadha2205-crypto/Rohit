import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
img = cv2.imread('tables.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to get a binary image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove small background noise using Morphological Opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Dilate the image to expand foreground boundaries
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Calculate distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Threshold the distance transform to get only the central peaks
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Unknown region is background minus foreground
unknown = cv2.subtract(sure_bg, sure_fg)

# Label individual foreground components
ret, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that background is 1 instead of 0
markers = markers + 1

# Explicitly mark the unknown boundary regions as 0
markers[unknown == 255] = 0

# Run Watershed algorithm
markers = cv2.watershed(img, markers)

# Draw boundaries on the original image in bright blue
img[markers == -1] = [255, 0, 0]

# Display the final result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("OpenCV Watershed Segmentation Results")
plt.axis('off')
plt.show()