import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets

# display images from train set

class DI():
    def __init__(self, root="gtsrb/Train"):
        self.dataset = datasets.ImageFolder(root=root)
        self.image_set = [s[0] for s in self.dataset.samples]
        self.label_set = [s[1] for s in self.dataset.samples]

    def count_img(self):
        classes, counts = np.unique(self.label_set, return_counts=True)
        for cls, count in zip(classes, counts):
            print(f"Class {cls}: {count} ")

    def show_random_images(self):
        labels = np.asarray(self.label_set)
        classes = np.unique(labels)

        indices = []
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            indices.append(np.random.choice(cls_indices))

        n = len(classes)
        ncols = 8
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
        axes = np.array(axes).reshape(-1)

        for ax, idx in zip(axes, indices):
            img = plt.imread(self.image_set[idx])
            ax.imshow(img)
            ax.set_title(f"Label:{self.label_set[idx]}")
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()