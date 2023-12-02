import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from segmentation_models import Unet
from sklearn.model_selection import train_test_split

from src.helper import get_checkpoint_path, get_data_dirs
from src.WSNET.helper import CreatePatches, generate_data, merge_patches, putall


def get_unet_model(train_model=False):
    sm.framework()

    sm.set_framework("tf.keras")

    in1 = tf.keras.Input(shape=(192, 192, 3))
    in2 = tf.keras.Input(shape=(192, 192, 3))
    layer = CreatePatches(64)
    layer = layer(in1)

    local_model = Unet(
        backbone_name="densenet169",
        input_shape=(64, 64, 3),
        classes=1,
        activation="sigmoid",
        encoder_freeze=False,
    )

    out0 = local_model(layer[0])
    out1 = local_model(layer[1])
    out2 = local_model(layer[2])
    out3 = local_model(layer[3])
    out4 = local_model(layer[4])
    out5 = local_model(layer[5])
    out6 = local_model(layer[6])
    out7 = local_model(layer[7])
    out8 = local_model(layer[8])

    X_patch = tf.keras.layers.Lambda(putall, arguments=dict(layer_count=9))(
        [out0, out1, out2, out3, out4, out5, out6, out7, out8]
    )

    X_patch = tf.keras.layers.Lambda(merge_patches)(X_patch)

    global_model = Unet(
        backbone_name="densenet169",
        input_shape=(192, 192, 3),
        classes=1,
        activation="sigmoid",
        encoder_freeze=False,
    )

    X_global_output = global_model(in2)

    X_final = tf.keras.layers.Concatenate(axis=3)([X_patch, X_global_output])
    X_final = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(X_final)

    model_1 = tf.keras.models.Model(inputs=[in1, in2], outputs=X_final)
    model_1.summary()

    data_dir, mask_dir = get_data_dirs(False)

    all_images = os.listdir(data_dir)

    to_train = 1  # ratio of number of train set images to use
    total_train_images = all_images[: int(len(all_images) * to_train)]
    len(total_train_images)

    train_images, validation_images = train_test_split(
        total_train_images, train_size=0.8, test_size=0.2, random_state=0
    )
    print(len(train_images), len(validation_images))

    BATCH_SIZE = 16
    width = 192
    height = 192

    train_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[train_images, BATCH_SIZE, (width, height), True, False, False, True],
        output_signature=(
            (tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3)), tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3))),
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 1)),
        ),
    )
    val_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[train_images, BATCH_SIZE, (width, height), False, True, False, True],
        output_signature=(
            (tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3)), tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3))),
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 1)),
        ),
    )

    epochs = 100
    checkpoint_path = get_checkpoint_path("unet", False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, save_best_only=True, mode="min"
        )
    ]
    model_1.compile(
        "Adam",
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), "binary_accuracy"],
    )

    if train_model:
        model_1.fit(
            train_gen,
            steps_per_epoch=np.ceil(float(len(train_images)) / float(BATCH_SIZE)),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)),
            verbose=1,
        )
    else:
        model_1.load_weights(checkpoint_path)

    # results = model_1.evaluate(val_gen, steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)))

    # print(results)
    return model_1, train_gen, val_gen
