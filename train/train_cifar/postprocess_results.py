import glob
import os

import numpy as np

from matplotlib import pyplot as plt


files = [file for file in glob.glob('results/*.npy') if 'cifar10' in file]

for file in files:
    results = np.load(file, allow_pickle=True).item()
    label = os.path.basename(file)[7: -4]
    plt.plot(results['val_acc'], label=label)
    print(label, 'best_acc:', np.max(results['val_acc']))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.show()
