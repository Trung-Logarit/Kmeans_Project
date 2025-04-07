import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans(image, k=3, max_iterations=10, tolerance=1e-4, seed=42):
    np.random.seed(seed)
    h, w, c = image.shape
    data = image.reshape((-1, 3)).astype(np.float32)

    # K-Means initialization (random centroid selection)
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]

    for iteration in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])
        
        if np.abs(centroids - new_centroids).sum() < tolerance:
            break
        centroids = new_centroids

        # Display step after each iteration
        if iteration == 9:  # After 10 iterations
            segmented_image = np.zeros_like(data)
            for i in range(k):
                segmented_image[cluster_assignments == i] = centroids[i]
            segmented_image = segmented_image.reshape((h, w, 3))
            plt.subplot(2, 5, 6)
            plt.imshow(cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.title(f"K-Means After 10 Iterations")
            plt.axis("off")

    cluster_assignments = cluster_assignments.reshape((h, w))
    return cluster_assignments, centroids, image

def kmeanspp(image, k=3, max_iterations=10, tolerance=1e-4, seed=42):
    np.random.seed(seed)
    h, w, c = image.shape
    data = image.reshape((-1, 3)).astype(np.float32)

    # K-Means++ initialization (smart centroid selection)
    first_centroid_index = 0
    centroids = [data[first_centroid_index]]

    for _ in range(k - 1):
        distances = np.sqrt(((data - np.array(centroids)[:, np.newaxis]) ** 2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        probabilities = min_distances / min_distances.sum()
        new_centroid = data[np.random.choice(data.shape[0], p=probabilities)]
        centroids.append(new_centroid)

    centroids = np.array(centroids)

    for iteration in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])
        
        if np.abs(centroids - new_centroids).sum() < tolerance:
            break
        centroids = new_centroids

        # Display step after each iteration
        if iteration == 9:  # After 10 iterations
            segmented_image = np.zeros_like(data)
            for i in range(k):
                segmented_image[cluster_assignments == i] = centroids[i]
            segmented_image = segmented_image.reshape((h, w, 3))
            plt.subplot(2, 5, 7)
            plt.imshow(cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.title(f"K-Means++ After 10 Iterations")
            plt.axis("off")

    cluster_assignments = cluster_assignments.reshape((h, w))
    return cluster_assignments, centroids, image

def plot_centroids_steps(image, k=3):
    # Step 1: K-Means and K-Means++ Initialization
    cluster_assignments_kmeans, centroids_kmeans, image_kmeans = kmeans(image, k)
    cluster_assignments_kmeanspp, centroids_kmeanspp, image_kmeanspp = kmeanspp(image, k)

    # Convert the image to RGB for visualization
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Step 2: Display all the steps side by side
    plt.figure(figsize=(20, 10))

    # Step 3: Original image
    plt.subplot(2, 5, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis("off")

    # Step 4: K-Means centroid initialization
    plt.subplot(2, 5, 2)
    plt.imshow(rgb_image)
    # Find closest pixels to centroids for visualization
    pixels = image.reshape((-1, 3))
    for i, cent in enumerate(centroids_kmeans):
        distances = np.sqrt(((pixels - cent)**2).sum(axis=1))
        closest_idx = np.argmin(distances)
        y, x = divmod(closest_idx, w)
        plt.scatter(x, y, color='yellow', marker='x', s=100, label="K-Means Centroids" if i==0 else "")
    plt.title("K-Means Centroids Initialization")
    plt.legend()
    plt.axis("off")

    # Step 5: K-Means++ centroid initialization
    plt.subplot(2, 5, 3)
    plt.imshow(rgb_image)
    for i, cent in enumerate(centroids_kmeanspp):
        distances = np.sqrt(((pixels - cent)**2).sum(axis=1))
        closest_idx = np.argmin(distances)
        y, x = divmod(closest_idx, w)
        plt.scatter(x, y, color='red', marker='o', s=100, label="K-Means++ Centroids" if i==0 else "")
    plt.title("K-Means++ Centroids Initialization")
    plt.legend()
    plt.axis("off")

    # Step 6: After 10 iterations of K-Means
    plt.subplot(2, 5, 4)
    segmented_kmeans = np.zeros_like(image)
    for i in range(k):
        segmented_kmeans[cluster_assignments_kmeans == i] = centroids_kmeans[i]
    plt.imshow(cv2.cvtColor(segmented_kmeans.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("After 10 Iterations (K-Means)")
    plt.axis("off")

    # Step 7: After 10 iterations of K-Means++
    plt.subplot(2, 5, 5)
    segmented_kmeanspp = np.zeros_like(image)
    for i in range(k):
        segmented_kmeanspp[cluster_assignments_kmeanspp == i] = centroids_kmeanspp[i]
    plt.imshow(cv2.cvtColor(segmented_kmeanspp.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("After 10 Iterations (K-Means++)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"  # Update your image path
image = cv2.imread(image_path)
if image is not None:
    plot_centroids_steps(image, k=3)
else:
    print("Error: Could not load image")