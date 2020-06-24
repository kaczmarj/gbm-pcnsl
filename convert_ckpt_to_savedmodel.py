"""
Convert the best checkpoints to TensorFlow SavedModel format.

Inference should be done with the SavedModel format.
"""

import os
import subprocess
import sys
import tempfile

PIP_TMP_TARGET = tempfile.mkdtemp(prefix="gbm-pcnsl-tmp-pip-target")
sys.path.insert(0, PIP_TMP_TARGET)


def pip_install(package):
    """Install package with `pip`."""
    cmd = f"{sys.executable} -m pip install --no-cache-dir --no-deps --target {PIP_TMP_TARGET} {package}"
    subprocess.check_call(cmd.split())


# Install Keras-applications, which has the Keras implementation of EfficientNet.
print("++ Installing Keras Applications package\n")
pip_install(
    "https://github.com/keras-team/keras-applications/tarball/bc89834ed36935ab4a4994446e34ff81c0d8e1b7"
)


from keras_applications import efficientnet
import tensorflow as tf
from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export

tfk = tf.keras
tfkl = tfk.layers


@keras_export(
    "keras.applications.efficientnet.EfficientNetB4",
    "keras.applications.EfficientNetB4",
)
@keras_modules_injection
def EfficientNetB4(*args, **kwargs):
    return efficientnet.EfficientNetB4(*args, **kwargs)


def get_model():
    """Return `tf.keras` model of EfficientNetB4 with output of 2."""
    base_model = EfficientNetB4(
        include_top=False, input_shape=(380, 380, 3), weights="imagenet"
    )
    base_model.activity_regularizer = tf.keras.regularizers.l2(l=0.01)

    _x = tfkl.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    _x = tfkl.Dropout(0.5)(_x)
    _x = tfkl.Dense(
        1,
        activation="sigmoid",
        name="predictions",
        kernel_initializer=efficientnet.DENSE_KERNEL_INITIALIZER,
    )(_x)
    model = tf.keras.Model(inputs=base_model.input, outputs=_x)
    return model


mapping = {
    "checkpoints/efficientnetb4-no-augment/ckpt_273_0.0000": "savedmodels/no-augment",
    "checkpoints/efficientnetb4-augment/ckpt_213_0.0001": "savedmodels/augment",
    "checkpoints/efficientnetb4-augment-noise/ckpt_203_0.0000": "savedmodels/augment-noise",
}

for ckpt, outpath in mapping.items():
    print("\n++ Creating", outpath)
    os.makedirs(outpath, exist_ok=False)
    model = get_model()
    model.load_weights(ckpt)
    model.save(outpath)

print("\n++++ Finished ++++")
