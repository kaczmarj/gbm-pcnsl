# Install newer keras_applications.
import subprocess
import sys
import time

_cmd = "pip install --no-cache-dir --no-deps --target=/tmp/python3jk https://github.com/keras-team/keras-applications/tarball/bc89834ed36935ab4a4994446e34ff81c0d8e1b7"
subprocess.run(_cmd.split(), check=True)
time.sleep(5)

sys.path.insert(0, "/tmp/python3jk")
del sys

from keras_applications import efficientnet

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.applications.efficientnet.EfficientNetB4",
    "keras.applications.EfficientNetB4",
)
@keras_modules_injection
def EfficientNetB4(*args, **kwargs):
    return efficientnet.EfficientNetB4(*args, **kwargs)


from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


def load_data(filename, size="224_224"):
    s = size
    with h5py.File(filename, "r") as f:
        x_gbm = f[f"/gbm/{s}/features"][:]
        y_gbm = f[f"/gbm/{s}/labels"][:]
        x_pcnsl = f[f"/pcnsl/{s}/features"][:]
        y_pcnsl = f[f"/pcnsl/{s}/labels"][:]

    # Transform from range [0, 1] to range [-1, 1].
    x_pcnsl *= 2.0
    x_pcnsl -= 1.0
    x_gbm *= 2.0
    x_gbm -= 1.0

    print("gbm features shape", x_gbm.shape)
    print("gbm labels shape", y_gbm.shape)
    print("pcnsl features shape", x_pcnsl.shape)
    print("pcnsl labels shape", y_pcnsl.shape)

    x = np.concatenate((x_gbm, x_pcnsl))
    y = np.concatenate((y_gbm, y_pcnsl)).astype(np.uint8)
    print("++ Unique labels", np.unique(y))
    print("++ Labels shape", y.shape)

    shuffle_inds = np.arange(y.shape[0])
    prng = np.random.RandomState(42)

    prng.shuffle(shuffle_inds)
    x = x[shuffle_inds]
    y = y[shuffle_inds]

    inds = prng.choice([0, 1], size=y.shape[0], p=[0.85, 0.15])
    x_train, y_train = x[inds == 0], y[inds == 0]
    x_val, y_val = x[inds == 1], y[inds == 1]

    print("++ Train shape:", x_train.shape, y_train.shape)
    print("++ Val Shape:", x_val.shape, y_val.shape)

    return (x_train, y_train), (x_val, y_val)


def get_datasets(filename, size="224_224", batch_size=32):
    (x_train, y_train), (x_val, y_val) = load_data(filename, size=size)

    n_train = y_train.shape[0] // batch_size
    n_val = y_val.shape[0] // batch_size

    def augment(x, y):
        x = tf.image.random_brightness(x, max_delta=2)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_hue(x, max_delta=0.25)
        return x, y

    d_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # d_train = d_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d_train = d_train.shuffle(1000, reshuffle_each_iteration=True)
    d_train = d_train.batch(batch_size, drop_remainder=True)
    d_train = d_train.repeat()

    d_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    d_val = d_val.batch(batch_size, drop_remainder=True)
    d_val = d_val.repeat()

    return (d_train, n_train), (d_val, n_val)


def get_model():
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


def train(prefix):
    (d_train, n_train), (d_val, n_val) = get_datasets(
        "data.h5", size="380_380", batch_size=32
    )

    def scheduler(epoch):
        import math

        if epoch < 50:
            return 1e-04
        else:
            return 1e-04 * math.exp(0.015 * (50 - epoch))

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        model = get_model()

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.Adam(1e-04)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(prefix / "ckpt_{epoch:03d}_{val_loss:0.4f}"),
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    h = model.fit(
        x=d_train,
        validation_data=d_val,
        steps_per_epoch=n_train,
        validation_steps=n_val,
        initial_epoch=0,
        epochs=300,
        callbacks=callbacks,
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        raise ValueError("missing save directory")
    prefix = sys.argv[1]
    prefix = Path(prefix)
    print("++ prefix:", prefix, flush=True)
    train(prefix)
