"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm





"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

        self.name = 'multifeature_'+opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.feature_dim = self.model.last_linear.in_features
        out_dict = nn.ModuleDict()
        for mode in opt.diva_features:
            out_dict[mode] = torch.nn.Linear(self.feature_dim, opt.embed_dim)
        self.has_merged = False

        self.model.last_linear  = out_dict

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        
    def merge_branches(self, weights=None):
        from IPython import embed; embed()
        if weights is None:
            pass
        else:
            pass
        self.has_merged = True

    def forward(self, x, warmup=False, **kwargs):
        z_dict = {}
        if warmup:
            with torch.no_grad():
                x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
                for layerblock in self.layer_blocks:
                    x = layerblock(x)
                prepool_y = x
                y = nn.functional.avg_pool2d(x, kernel_size=x.shape[2])
                y = y.view(y.size(0),-1)
        else:
            x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
            for layerblock in self.layer_blocks:
                x = layerblock(x)
            prepool_y = x
            y = nn.functional.avg_pool2d(x, kernel_size=x.shape[2])
            y = y.view(y.size(0),-1)

        for key,embed in self.model.last_linear.items():
            z = embed(y)
            if 'normalize' in self.pars.arch:
                z = torch.nn.functional.normalize(z, dim=-1)
            z_dict[key] = z

        return {'embeds':z_dict, 'avg_features':y, 'features':x, 'extra_embeds': prepool_y}
