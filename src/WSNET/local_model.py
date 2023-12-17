import tensorflow as tf
from segmentation_models import FPN, Linknet, PSPNet, Unet

from src.WSNET.helper import CreatePatches, putconcate, putconcate_vert


def create_local_model(segmentation_model, input_size=192, activation_function="sigmoid"):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet"
    :param input_size: width and height of input images, inputs must be square image. Default is 192. Must be dividable
    by 48 and 64
    """
    if input_size % 48 != 0:
        raise ValueError("Input size must be dividable by 48 and 64")

    if input_size % 64 != 0:
        raise ValueError("Input size must be dividable by 48 and 64")

    input = tf.keras.Input(shape=(input_size, input_size, 3))

    if segmentation_model == "fpn":
        # input size must be dividable by 32
        patch_size = 64
        local_model = FPN(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "pspnet":
        patch_size = 48
        local_model = PSPNet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "linknet":
        # input size must be dividable by 32
        patch_size = 64
        local_model = Linknet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "unet":
        # input size must be dividable by 32
        patch_size = 64
        local_model = Unet(
            backbone_name="mobilenet",
            input_shape=(patch_size, patch_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    else:
        raise ValueError('Parameter segmentation_model must be one of "fpn", "pspnet", "linknet", "unet"')

    layer = CreatePatches(patch_size)
    layer = layer(input)

    # put layers horizontally back together
    patch_outputs = []
    patches_per_dimension = int((input_size / patch_size))
    for i in range(0, patches_per_dimension * patches_per_dimension):
        out = local_model(layer[i])
        patch_outputs.append(out)

    horizontally_concat = []
    for i in range(0, patches_per_dimension):
        patch = tf.keras.layers.Lambda(putconcate, arguments=dict(layer_count=patches_per_dimension))(
            patch_outputs[i * patches_per_dimension : (i + 1) * patches_per_dimension]
        )
        horizontally_concat.append(patch)

    x_patch = tf.keras.layers.Lambda(putconcate_vert, arguments=dict(layer_count=patches_per_dimension))(
        horizontally_concat
    )

    x_final = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x_patch)

    model = tf.keras.models.Model(inputs=[input], outputs=x_final)

    return model
