import cv2, numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
img = cv2.imread(list(uploaded.keys())[0])

if img is None:
    print("Error: Image not found")
else:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)
    dilate = cv2.dilate(closing, kernel, 3)

    titles = ["Original","Gray","Threshold","Closing","Dilation"]
    imgs = [rgb, gray, thresh, closing, dilate]

    plt.figure(figsize=(12,8))
    for i,(t,im) in enumerate(zip(titles,imgs),1):
        plt.subplot(2,3,i)
        plt.imshow(im, cmap='gray' if i>1 else None)
        plt.title(t); plt.axis('off')

    plt.tight_layout(); plt.show()
    plt.imsave('dilation.png', dilate)
