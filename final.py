import cv2
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

# Load the two input images
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

# Define the scale to resize the images
scale = 0.5

# Resize the images
img1 = cv2.resize(img1, (0,0), fx=scale, fy=scale)
img2 = cv2.resize(img2, (0,0), fx=scale, fy=scale)

# Convert the images to YUV color space
img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
y1, u1, v1 = cv2.split(img1_yuv)
img1_y = y1+u1+v1
img1_y_eq = cv2.equalizeHist(img1_y)
img1_y_blur = cv2.GaussianBlur(img1_y_eq, (3,3), 0)

img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
y2, u2, v2 = cv2.split(img2_yuv)
img2_y = y2+u2+v2
img2_y_eq = cv2.equalizeHist(img2_y)
img2_y_blur = cv2.GaussianBlur(img2_y_eq, (3, 3), 0)

# Display the YUV converted images
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(cv2.cvtColor(img1_yuv, cv2.COLOR_YUV2RGB))
axs[0, 0].set_title('Imagem 1 (YUV)')
axs[0, 1].imshow(cv2.cvtColor(img2_yuv, cv2.COLOR_YUV2RGB))
axs[0, 1].set_title('Imagem 2 (YUV)')
axs[1, 0].imshow(img1_y_blur, cmap='gray')
axs[1, 0].set_title('Imagem 1 Y (Eq e Gauss)')
axs[1, 1].imshow(img2_y_blur, cmap='gray')
axs[1, 1].set_title('Imagem 2 Y (Eq e Gauss)')
plt.show()

# Create a SIFT object
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the two images
kp1, des1 = sift.detectAndCompute(img1_y_blur, None)
kp2, des2 = sift.detectAndCompute(img2_y_blur, None)

# Find matches between the descriptors in the two images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Select the best matches
n_matches = 1000
matches = matches[:n_matches]

# Extract the interest points from the matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

plt.subplot(2,2,3)
plt.imshow(cv2.drawKeypoints(img1, kp1, None, color=(0,255,0)))
plt.title('Imagem 1 - Features')

plt.subplot(2,2,4)
plt.imshow(cv2.drawKeypoints(img2, kp2, None, color=(0,255,0)))
plt.title('Imagem 2 - Features')

plt.show()


tri = Delaunay(pts1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax.triplot(pts1[:, 0], pts1[:, 1], tri.simplices, color='b', alpha=0.9)
ax.set_title('Imagem 1 com Delaunay')
plt.show()

# Calculate affine homographies from the Delaunay triangles and keypoints
homographies = []
H, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC)
homographies.append(H)

print(homographies)

# Use the homography to transform points from one image to the other

# Check if homographies list is not empty
if homographies:
    # Apply the first homography to img1
    img1_homography = cv2.warpAffine(img1, homographies[0], (img2.shape[1], img2.shape[0]))

    plt.imshow(cv2.cvtColor(img1_homography, cv2.COLOR_YUV2RGB))
    plt.title('Imagem 1 Transformada')
    plt.show()

# Find keypoints and descriptors in the two images
kp1, des1 = sift.detectAndCompute(img1_homography, None)

# Find matches between the descriptors in the two images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Select the best matches
n_matches = 1000
matches = matches[:n_matches]

# Extract the interest points from the matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])

all_pts_transformed = np.vstack((pts1, pts2)).reshape(-1, 2)
k = 4
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(all_pts_transformed)

colors = ['red', 'blue', 'green', 'purple']
for i in range(k):
    cluster_pts = all_pts_transformed[kmeans.labels_ == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], c=colors[i], alpha=0.5)

plt.xlim(0, img1.shape[1] + img2.shape[1])
plt.ylim(img1.shape[0], 0)
plt.title('K-means')
plt.show()

# Get the labels for each point
labels = kmeans.labels_

# Plot the segmented planes on top of the original image
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(cv2.cvtColor(img1_homography, cv2.COLOR_BGR2RGB))

# Define colors for the planes
colors = ['r', 'g', 'b', 'y']

# Iterate over each cluster and plot the corresponding points with a different color
for i in range(k):
    mask = labels == i
    ax.scatter(all_pts_transformed[:, 0][mask], all_pts_transformed[:, 1][mask], c=colors[i], s=3)

plt.title('Identificação dos planos')
plt.show()

# Show the plot
plt.show()