from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.segmentation import mark_boundaries

# Load the image
image = imread(filedialog.askopenfilename())

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Perform K-Means clustering
num_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(pixels)

# Get the labels assigned to each pixel
labels = kmeans.labels_

# Reshape the labels back to the original image shape
segmented_image = labels.reshape(image.shape[:2])

# Get the centroid colors of each cluster
centroids = kmeans.cluster_centers_

# Create a new image to mark the objects
object_image = np.zeros_like(image)

# Mark the objects based on color
for i in range(num_clusters):
    object_image[segmented_image == i] = centroids[i]

# Visualize the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(segmented_image, cmap='nipy_spectral')
plt.title('Segmented Image')

plt.subplot(1, 3, 3)
plt.imshow(object_image)
plt.title('Objects Marked by Color')

plt.show()