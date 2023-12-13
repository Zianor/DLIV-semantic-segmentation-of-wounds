import tensorflow as tf
from segmentation_models import FPN, Linknet, PSPNet, Unet

from src.WSNET.helper import CreatePatches, merge_patches, putall


def create_global_local_model(segmentation_model, input_size=192):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet"
    :param input_size: width and height of input images, inputs must be square image. Default is 192. Must be dividable
    by 48 and 64
    """
    if input_size % 48 != 0:
        raise ValueError("Input size must be dividable by 48 and 64")

    if input_size % 64 != 0:
        raise ValueError("Input size must be dividable by 48 and 64")

    local_input = tf.keras.Input(shape=(input_size, input_size, 3))
    global_input = tf.keras.Input(shape=(input_size, input_size, 3))

    if segmentation_model == "fpn":
        # input size must be dividable by 32
        patch_size = 64
        local_model = FPN(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
        global_model = FPN(
            backbone_name="mobilenet",
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
    elif segmentation_model == "pspnet":
        patch_size = 48
        local_model = PSPNet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
        global_model = PSPNet(
            backbone_name="mobilenet",
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
    elif segmentation_model == "linknet":
        # input size must be dividable by 32
        patch_size = 64
        local_model = Linknet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
        global_model = Linknet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
    elif segmentation_model == "unet":
        # input size must be dividable by 32
        patch_size = 64
        local_model = Unet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
        global_model = Unet(
            backbone_name="mobilenet",
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation="sigmoid",
            encoder_freeze=False,
        )
    else:
        raise ValueError('Parameter segmentation_model must be one of "fpn", "pspnet", "linknet", "unet"')

    local_layer = CreatePatches(patch_size)
    local_layer = local_layer(local_input)
    # create output for all patches
    patch_outputs = []
    number_patches = int((input_size / patch_size) * (input_size / patch_size))
    for i in range(0, number_patches):
        out = local_model(local_layer[i])
        patch_outputs.append(out)

    x_patch = tf.keras.layers.Lambda(putall, arguments=dict(layer_count=number_patches))(patch_outputs)

    x_patch = tf.keras.layers.Lambda(merge_patches)(x_patch)

    x_global_output = global_model(global_input)

    x_final = tf.keras.layers.Concatenate(axis=3)([x_patch, x_global_output])

    # final convolution of global and local output
    x_final = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x_final)

    model = tf.keras.models.Model(inputs=[local_input, global_input], outputs=x_final)

    return model
