"""
Give the prediction probability that an MR image contains glioblastoma (GBM)
or primary central nervous system lymphoma (PCNSL).

Images must be axial, T1-weighted, contrast-enhanced MR images.

Usage: python predict.py MODELPATH IMAGEPATH...
"""

import os
from pathlib import Path
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

REQUIRED_IMAGE_SIZE = (380, 380)


def get_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    # Transform to [0, 1].
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, REQUIRED_IMAGE_SIZE)
    # Transform to [-1, 1].
    img *= 2
    img -= 1
    return img


def predict_on_images(file_paths):
    model = tf.keras.models.load_model(model_path, compile=False)
    s = "{: <40} {:.02f} % {}"
    for file_path in file_paths:
        img = get_image(file_path)
        prob_pcnsl = model.predict(img[None])
        prob_pcnsl = float(prob_pcnsl[0])
        prob_gbm = 1 - prob_pcnsl
        if prob_pcnsl > prob_gbm:
            print(s.format(file_path, prob_pcnsl * 100, "PCNSL"))
        else:
            print(s.format(file_path, prob_gbm * 100, "GBM"))


def main(model_path, file_paths):

    if not Path(model_path).exists():
        raise FileNotFoundError("model path not found: {}".format(model_path))
    for file_path in file_paths:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                "image file not found: {}".format(file_path)
            )

    predict_on_images(file_paths)


if __name__ == "__main__":

    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)
    elif len(sys.argv) < 3:
        print("usage: python {} MODELPATH IMAGEPATH...".format(sys.argv[0]))
        sys.exit(1)
    model_path = sys.argv[1]
    file_paths = sys.argv[2:]

    main(model_path, file_paths)
