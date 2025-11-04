#TASK-1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

file_id = "1u9yz-Kslh-zkCLmFuKD3f8sFK8j179LA"
url = f"https://drive.google.com/uc?id={file_id}"

# Read image
image = io.imread(url)

# Show original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixels = image.reshape(-1, 3)

# Apply KMeans clustering
k = 7
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(pixels)

# Build segmented image
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape).astype(np.uint8)

# Display segmented image
plt.imshow(segmented_img)
plt.title(f'Segmented Image with {k} colors')
plt.axis('off')
plt.show()

#TASK-2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

file_id = "1u9yz-Kslh-zkCLmFuKD3f8sFK8j179LA"
url = f"https://drive.google.com/uc?id={file_id}"

# Read image
image = io.imread(url)

# Show original image
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

pixels = image.reshape(-1, 3)


distortions = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    distortions.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(6,4))
plt.grid(True)
plt.plot(K, distortions, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Distortion)')
plt.title('Elbow Method For Optimal k')
plt.show()

optimal_k = 7

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(pixels)


segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape).astype(np.uint8)


plt.figure(figsize=(12,6))

# Original
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Segmented
plt.subplot(1,2,2)
plt.imshow(segmented_img)
plt.title(f"Segmented Image with {optimal_k} colors")
plt.axis("off")

plt.show()
