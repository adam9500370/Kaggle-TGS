import torchvision.models as models

from ptsemseg.models.pspnet import *


def get_model(name, n_classes, version=None, f_scale=1):
    model = _get_model_instance(name)

    if name == 'pspnet':
        model = model(n_classes=n_classes, version=version, f_scale=f_scale)

    return model

def _get_model_instance(name):
    try:
        return {
            'pspnet': pspnet,
        }[name]
    except:
        print('Model {} not available'.format(name))
