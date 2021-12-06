import numpy as np
import os
from pathlib import Path

folder = Path('/home/sancha/data/kitti_detection/training')

anno_file_paths = [folder / 'label_2' / file for file in os.listdir(folder / 'label_2')]

data = []
for anno_file_path in anno_file_paths:
    file_data = np.atleast_2d(np.loadtxt(anno_file_path, dtype=str))
    file_data = file_data[:, [0, 9, 10]]  # keep only class name and wl
    file_data = file_data[(file_data[:, 0] != 'DontCare') & (file_data[:, 0] != 'Misc')]  # remove the DontCare class
    data.append(file_data)

data = np.vstack(data)
class_names = set(data[:, 0])

with open("analysis/kitti_sizes.txt", "w") as f:
    for class_name in class_names:
        class_data = data[data[:, 0] == class_name, :]
        class_data = class_data[:, 1:].astype(np.float32)
        w, h = class_data.mean(axis=0)
        print(f"{class_name} {w} {h} {len(class_data)}", file=f)
