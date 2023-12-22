import tensorflow as tf
from segmentation_models import FPN, Linknet, PSPNet, Unet


def create_global_global_model(
    segmentation_model_1,
    segmentation_model_2,
    input_size=192,
    backbone="mobilenet",
    activation_function="sigmoid",
):
    """This architecture consists of two segmentation models. The resulting segmentations are combined by a 1x1
    convolution to obtain the final segmentation

    :param segmentation_model_1: one of "fpn", "pspnet", "linknet", "unet"
    :param segmentation_model_2: one of "fpn", "pspnet", "linknet", "unet", should be different from
    segmentation_model_1
    :param input_size: width and height of input images, inputs must be square image. Default is 192. Must be dividable
    by 48 and 64
    :param backbone: name of the backbone that should be used, default is mobilenet
    :param activation_function: activation function used, default is sigmoid
    """
    input1 = tf.keras.Input(shape=(input_size, input_size, 3))
    input2 = tf.keras.Input(shape=(input_size, input_size, 3))

    if segmentation_model_1 == segmentation_model_2:
        raise ValueError(
            "Parameter segmentation_model_1 should be different from parameter segmentation_model_2"
        )

    if segmentation_model_1 == "fpn":
        model_1 = FPN(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_1 == "pspnet":
        model_1 = PSPNet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_1 == "linknet":
        model_1 = Linknet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_1 == "unet":
        model_1 = Unet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    else:
        raise ValueError('Parameter segmentation_model_1 must be one of "fpn", "pspnet", "linknet", "unet"')

    if segmentation_model_2 == "fpn":
        # input size must be dividable by 32
        model_2 = FPN(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_2 == "pspnet":
        model_2 = PSPNet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_2 == "linknet":
        # input size must be dividable by 32
        model_2 = Linknet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    elif segmentation_model_2 == "unet":
        model_2 = Unet(
            backbone_name=backbone,
            input_shape=(input_size, input_size, 3),
            classes=1,
            activation=activation_function,
            encoder_freeze=False,
        )
    else:
        raise ValueError('Parameter segmentation_model_2 must be one of "fpn", "pspnet", "linknet", "unet"')

    x_model1_output = model_1(input1)
    x_model2_output = model_2(input2)

    x_final = tf.keras.layers.Concatenate(axis=3)([x_model1_output, x_model2_output])

    # final convolution of global and local output
    x_final = tf.keras.layers.Conv2D(1, 1, activation=activation_function)(x_final)

    model = tf.keras.models.Model(inputs=[input1, input2], outputs=x_final)

    return model
