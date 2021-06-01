import architectures.resnet50
import architectures.resnext101
import architectures.bninception
import architectures.multifeature_resnet18
import architectures.multifeature_resnet50
import architectures.multifeature_resnet101
import architectures.multifeature_bninception
import architectures.multifeature_bit
import architectures.multifeature_efficientb0
import architectures.resnet18
import architectures.resnet101
import architectures.bit
import architectures.efficientb0

def select(arch, opt):
    if  'multifeature_resnet50' in arch:
        return multifeature_resnet50.Network(opt)
    if  'multifeature_resnet18' in arch:
        return multifeature_resnet18.Network(opt)
    if  'multifeature_resnet101' in arch:
        return multifeature_resnet101.Network(opt)
    if  'multifeature_bninception' in arch:
        return multifeature_bninception.Network(opt)
    if  'multifeature_bit' in arch:
        return multifeature_bit.Network(opt)
    if  'multifeature_efficientb0' in arch:
        return multifeature_efficientb0.Network(opt)
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'resnet18' in arch:
        return resnet18.Network(opt)
    if 'resnet101' in arch:
        return resnet101.Network(opt)
    if 'resnext101' in arch:
        return resnext101.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
    if 'bit' in arch:
        return bit.Network(opt)
    if 'efficientb0' in arch:
        return efficientb0.Network(opt)
