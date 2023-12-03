import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import os

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from segmentation_models import PSPNet
from sklearn.model_selection import train_test_split

from src.helper import get_checkpoint_path, get_data_dirs
from src.WSNET.helper import CreatePatches, generate_data, putconcate, putconcate_vert


def get_pspnet_local_model(train_model=False):
    sm.framework()
    sm.set_framework("tf.keras")
    # sample_image = np.random.rand(1, 192 , 192 , 3 ).astype(np.float32)
    in1 = tf.keras.Input(shape=(192, 192, 3))
    in2 = tf.keras.Input(shape=(192, 192, 3))
    # input = (Input(shape=(192, 192, 3), name='input'))
    layer = CreatePatches(48)
    # print(layer)
    layer = layer(in1)

    local_model = PSPNet(backbone_name="mobilenet", input_shape=(48, 48, 3), classes=1, activation="sigmoid")

    out0 = local_model(layer[0])
    out1 = local_model(layer[1])
    out2 = local_model(layer[2])
    out3 = local_model(layer[3])
    out4 = local_model(layer[4])
    out5 = local_model(layer[5])
    out6 = local_model(layer[6])
    out7 = local_model(layer[7])
    out8 = local_model(layer[8])
    out9 = local_model(layer[9])
    out10 = local_model(layer[10])
    out11 = local_model(layer[11])
    out12 = local_model(layer[12])
    out13 = local_model(layer[13])
    out14 = local_model(layer[14])
    out15 = local_model(layer[15])

    X_patch1 = tf.keras.layers.Lambda(putconcate)([out0, out1, out2, out3])
    X_patch2 = tf.keras.layers.Lambda(putconcate)([out4, out5, out6, out7])
    X_patch3 = tf.keras.layers.Lambda(putconcate)([out8, out9, out10, out11])
    X_patch4 = tf.keras.layers.Lambda(putconcate)([out12, out13, out14, out15])

    X_patch = tf.keras.layers.Lambda(putconcate_vert)([X_patch1, X_patch2, X_patch3, X_patch4])

    # In[16]:

    # In[17]:

    # out_combined = tf.stack([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15], axis=1)

    # In[18]:

    # rec_new = tf.space_to_depth(X_patch[-1],4)
    # rec_new = tf.reshape(rec_new,[-1,192,192,1])

    # In[19]:

    # global_model = PSPNet(backbone_name='mobilenet',input_shape=(192, 192, 3),classes=1,activation='sigmoid')

    # In[20]:

    # in2 = tf.keras.Input(shape=(192,192,3))
    # X_global_output = global_model(in2)

    # is this the last convolution between local and global? but where is the gobal model
    X_final = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(X_patch)

    model_1 = tf.keras.models.Model(inputs=[in1], outputs=X_final)
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
        args=[train_images, BATCH_SIZE, (width, height), True, False, False],
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3)),
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 1)),
        ),
    )
    val_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[validation_images, BATCH_SIZE, (width, height), False, True, False],
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 3)),
            tf.TensorSpec(shape=(BATCH_SIZE, 192, 192, 1)),
        ),
    )
    for layer in model_1.layers:
        layer.trainable = True

    epochs = 100

    checkpoint_path = get_checkpoint_path("pspnet_local")

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


get_pspnet_local_model()