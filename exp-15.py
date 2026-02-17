import cv2, numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()

# Read uploaded image (use uploaded file name)
img = cv2.imread(list(uploaded.keys())[0])
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = np.float32(rgb.reshape(-1,3))
_, labels, centers = cv2.kmeans(
    pixels, 3, None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
    10, cv2.KMEANS_RANDOM_CENTERS
)

segmented = np.uint8(centers)[labels.flatten()].reshape(rgb.shape)

plt.subplot(121); plt.imshow(rgb); plt.title("Original"); plt.axis('off')
plt.subplot(122); plt.imshow(segmented); plt.title("Segmented"); plt.axis('off')
plt.show()
