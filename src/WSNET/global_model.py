import tensorflow as tf
from segmentation_models import FPN, Linknet, PSPNet, Unet


def create_global_model(
    segmentation_model, input_size=192, backbone="mobilenet", activation_function="sigmoid"
):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet"
    :param input_size: width and height of input images, inputs must be square image. Default is 192
    :param backbone: name of the backbone that should be used, default is mobilenet
    :param activation_function: activation function used, default is sigmoid
    """
    input = tf.keras.Input(shape=(input_size, input_size, 3))

    if segmentation_model == "fpn":
        global_model = FPN(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "pspnet":
        global_model = PSPNet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "linknet":
        global_model = Linknet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model == "unet":
        global_model = Unet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    else:
        raise ValueError('Parameter segmentation_model must be one of "fpn", "pspnet", "linknet", "unet"')

    x_output = global_model(input)

    x_final = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x_output)

    model = tf.keras.models.Model(inputs=[input], outputs=x_final)

    return model
