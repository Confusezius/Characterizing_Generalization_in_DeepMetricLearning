"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import timm




"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = timm.create_model('efficientnet_b0', pretrained=True)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.classifier.in_features, opt.embed_dim)

        self.out_adjust = None
        self.extra_out  = None

        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)
        self.pool_aux  = torch.nn.AdaptiveMaxPool2d(1) if 'double' in opt.arch else None



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
