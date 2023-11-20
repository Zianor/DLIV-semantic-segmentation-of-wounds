import os

import numpy as np

os.environ["SM_FRAMEWORK"] = "tf.keras"


import os

import segmentation_models as sm
import tensorflow as tf
from segmentation_models import Linknet
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Conv2D, Lambda
from tensorflow.keras.models import Model

from src.helper import get_checkpoint_path, get_data_dirs
from src.WSNET.helper import CreatePatches, generate_data, merge_patches, putall

sm.framework()
sm.set_framework("tf.keras")


class StitchPatches(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super(StitchPatches, self).__init__()
        self.batch_size = batch_size

    def call(self, inputs):
        print(inputs)
        patches = []
        main_image = np.empty([inputs.shape[0], 192, 192, inputs.shape[3]])
        for k in range(0, inputs.shape[0], self.batch_size):
            for i in range(0, 192, 48):
                for j in range(0, 192, 48):
                    main_image[i : i + 48, j : j + 48, :] = inputs[k]
        return main_image


in1 = tf.keras.Input(shape=(192, 192, 3))
in2 = tf.keras.Input(shape=(192, 192, 3))
layer = CreatePatches(64)
layer = layer(in1)

local_model = Linknet(
    backbone_name="mobilenet", input_shape=(64, 64, 3), classes=1, activation="sigmoid", encoder_freeze=False
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

X_patch = Lambda(putall)([out0, out1, out2, out3, out4, out5, out6, out7, out8])
print(X_patch)

# out_combined = tf.stack([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15], axis=1)

X_patch = Lambda(merge_patches)(X_patch)

global_model = Linknet(
    backbone_name="mobilenet",
    input_shape=(192, 192, 3),
    classes=1,
    activation="sigmoid",
    encoder_freeze=False,
)

X_global_output = global_model(in2)


X_final = Concatenate(axis=3)([X_patch, X_global_output])
X_final = Conv2D(1, 1, activation="sigmoid")(X_final)

model_1 = Model(inputs=[in1, in2], outputs=X_final)
model_1.summary()

# data_dir = "./sample_corrected/sample/"
# #data_dir = "./train/sample/"
# #data_dir = "./miccai2/images/"
# mask_dir = "./sample_corrected/mask/"
# #mask_dir = "./train/masked_images/"
# #mask_dir = "./miccai2/labels/"
data_dir, mask_dir = get_data_dirs(False)

all_images = os.listdir(data_dir)

to_train = 1  # ratio of number of train set images to use
total_train_images = all_images[: int(len(all_images) * to_train)]
len(total_train_images)

# split train set and test set
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

# In[61]:
for layer in model_1.layers:
    layer.trainable = True

epochs = 100

checkpoint_path = get_checkpoint_path("linknet_wstech_imagenet1_nofreeze_densenet")

callbacks = [ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, mode="min")]
model_1.compile(
    "Adam",
    loss=sm.losses.DiceLoss(),
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), "binary_accuracy"],
)

model_1.fit(
    train_gen,
    steps_per_epoch=np.ceil(float(len(train_images)) / float(BATCH_SIZE)),
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_gen,
    validation_steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)),
    verbose=1,
)
