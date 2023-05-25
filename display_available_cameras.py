import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

imgs = []

for i in tqdm(range(16), desc="scanning devices"):
    cam = cv2.VideoCapture(i)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _, img = cam.read()
    imgs.append(img)

imgs = enumerate(imgs)
imgs = filter(lambda x: x[1] is not None, imgs)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for _, ax in np.ndenumerate(axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

for i, img in imgs:
    ax = axes[i // 4, i % 4]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), aspect="auto")
    ax.set_title(f"device {i}")

plt.show()
