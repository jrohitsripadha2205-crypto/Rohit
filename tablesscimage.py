import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

def execute_segmentation_pipeline(image_path, min_distance=15, compactness=0.1):
    """
    A unified CV pipeline utilizing OpenCV for preprocessing and 
    skimage for high-precision watershed segmentation.
    """
    # 1. Load Image and Convert to Grayscale (OpenCV)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Reduction (OpenCV)
    # Blurring removes minor surface textures that cause fake local peaks
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Binarization via Otsu's Thresholding (OpenCV)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Morphological Cleaning (OpenCV)
    # Opening removes small stray white noise spots in the background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 5. Exact Distance Transform (OpenCV)
    # Calculates exact distance from every foreground pixel to the background boundary
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    # 6. Peak Detection for Markers (skimage)
    # Finds local peaks in the distance map that are at least 'min_distance' apart
    coordinates = peak_local_max(dist_transform, min_distance=min_distance, labels=clean_mask)
    
    # Convert peaks to a boolean map matching our image layout
    peak_mask = np.zeros(dist_transform.shape, dtype=bool)
    peak_mask[tuple(coordinates.T)] = True
    
    # 7. Label Unique Seeds (scipy / skimage compatible)
    # Assigns a unique ID integer (1, 2, 3...) to each distinct peak/marker
    markers, _ = ndimage.label(peak_mask)
    
    # 8. Run final Watershed Segmentation (skimage)
    # We pass the inverted distance transform (-dist_transform) so peaks act as valleys.
    # The clean_mask ensures the flooding remains confined to our objects.
    final_labels = watershed(-dist_transform, markers, mask=clean_mask, compactness=compactness)
    
    return img, final_labels

# Quick Execution Test Context:
img, labels = execute_segmentation_pipeline('tables.png', min_distance=15, compactness=0.1)


import matplotlib.pyplot as plt

# Display the original image and the segmented labels
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
axes[0].set_title('Original Image')
axes[0].axis('off')

# For segmented labels, use a colormap to distinguish regions
axes[1].imshow(labels, cmap='nipy_spectral')
axes[1].set_title('Segmented Image (Watershed Labels)')
axes[1].axis('off')

plt.tight_layout()
plt.show()