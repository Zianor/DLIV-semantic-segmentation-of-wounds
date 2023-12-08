from WSNET.Medical_CNN_FPN import get_fpn_model
from WSNET.Medical_CNN_FPN_Local import get_fpn_local_model
from WSNET.Medical_CNN_Linknet import get_linknet_model
from WSNET.Medical_CNN_Linknet_Local import get_linknet_local_model
from WSNET.Medical_CNN_PSPNET import get_pspnet_model
from WSNET.Medical_CNN_PSPNET_local import get_pspnet_local_model
from WSNET.Medical_CNN_Unet import get_unet_model
from WSNET.Medical_CNN_Unet_Local import get_unet_local_model

if __name__ == "__main__":
    get_unet_model(True)
    get_pspnet_model(True)
    get_fpn_model(True)
    get_linknet_model(True)

    get_unet_local_model(True)
    get_pspnet_local_model(True)
    get_fpn_local_model(True)
    get_linknet_local_model(True)
