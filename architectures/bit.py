"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import timm
import torchvision



"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt

        self.name = opt.arch

        self.model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == timm.models.layers.norm_act.GroupNormAct, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.head.fc.in_channels, opt.embed_dim)

        self.out_adjust = None
        self.extra_out  = None

        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)
        self.pool_aux  = torch.nn.AdaptiveMaxPool2d(1) if 'double' in opt.arch else None

        self.specific_normalization = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    def forward(self, x, warmup=False, **kwargs):
        if warmup:
            with torch.no_grad():
                x = self.model.forward_features(x)
                prepool_y = x
                if self.pool_aux is not None:
                    y = self.pool_aux(x) + self.pool_base(x)
                else:
                    y = self.pool_base(x)
                y = y.view(y.size(0),-1)
                x,y,prepool_y = x.detach(), y.detach(), prepool_y.detach()
        else:
            x = self.model.forward_features(x)
            prepool_y = x
            if self.pool_aux is not None:
                y = self.pool_aux(x) + self.pool_base(x)
            else:
                y = self.pool_base(x)
            y = y.view(y.size(0),-1)

        z = self.model.last_linear(y)

        if 'normalize' in self.pars.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds':z, 'avg_features':y, 'features':x, 'extra_embeds': prepool_y}
