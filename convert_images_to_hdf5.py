"""
Convert image data to HDF5. Expects data to be in `./data/gbm` and `./data/pcnsl`.
"""

from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def _preproccess_one(filename, size):
    img = Image.open(filename)
    img = img.convert("RGB")
    img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.asarray(img) / 255.0
    return img.astype(np.float32)


def _preprocess(path, size, label):
    files = list(path.glob("*"))
    nfiles = len(files)
    x = np.zeros((nfiles, *size, 3), dtype=np.float32)
    print(f"0 / {nfiles}", end="\r")
    for j, f in enumerate(files):
        x[j] = _preproccess_one(f, size=size)
        print(f"{j + 1} / {nfiles}", end="\r")
    print()
    return x, np.zeros(nfiles, dtype=np.uint8) + label


def save_one_to_hdf5(fp, size):
    print(f"++ Saving size {size}")
    s = "_".join(map(str, size))
    x0, y0 = _preprocess(Path("data/gbm/"), size=size, label=0)
    x1, y1 = _preprocess(Path("data/pcnsl/"), size=size, label=1)
    with h5py.File(fp, mode="a") as f:
        f.create_dataset(f"/gbm/{s}/features", data=x0, compression="lzf")
        f.create_dataset(f"/gbm/{s}/labels", data=y0, compression="lzf")
        f.create_dataset(f"/pcnsl/{s}/features", data=x1, compression="lzf")
        f.create_dataset(f"/pcnsl/{s}/labels", data=y1, compression="lzf")


if __name__ == "__main__":
    sizes = [(224, 224), (300, 300), (380, 380), (600, 600)]
    for size in sizes:
        save_one_to_hdf5("data.h5", size=size)
