import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from thop import profile
from torch import nn
from config import opt


def test_ptflops():
    net = models.resnet50()
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def test_thop():
    model = models.resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def count_ptflops(model, inputs_dim, tag):
    macs, params = get_model_complexity_info(model, inputs_dim, as_strings=False, print_per_layer_stat=False, )
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


def count_thop(model, inputs_dim, tag):
    input = torch.randn((1,) + inputs_dim)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


class GEN_I(torch.nn.Module):
    def __init__(self, image_dim, hidden_dim, output_dim):
        super(GEN_I, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim

        self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
            )
        self.hash_module = nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh())

    def forward(self, x):
        f_x = self.image_module(x)
        x_code = self.hash_module(f_x).reshape(-1, self.output_dim)
        return x_code


class GEN_T(torch.nn.Module):
    def __init__(self, text_dim, hidden_dim, output_dim):
        super(GEN_T, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim
        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(True)
        )

        self.hash_module = nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh())

    def forward(self, y):
        f_y = self.text_module(y)
        y_code = self.hash_module(f_y).reshape(-1, self.output_dim)
        return y_code


class DIS_F(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hash_dim):
        super(DIS_F, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.feature_dis = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim//2, 1, bias=True)
        )

    def forward(self, f):
        f = self.feature_dis(f)
        return f


class DIS_H(torch.nn.Module):
    def __init__(self, hash_dim):
        super(DIS_H, self).__init__()
        self.hash_dim = hash_dim

        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1, bias=True)
        )

    def forward(self, f):
        h = self.hash_dis(f)
        return h


gi = GEN_I(512, opt.hidden_dim, 128)
gt = GEN_T(768, opt.hidden_dim, 128)
dh = DIS_H(128)
df = DIS_F(2048, 1024, 128)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

input_dims = (512,)
model = gi


def calculate_stats_for_unhd(method='ptflops'):
    if method == 'ptflops':
        f = count_ptflops
    else:
        f = count_thop

    print('\n\n\n' + method + '\n')
    print('Module stats:')
    macsgi, paramsgi = f(gi, (512,), 'img')
    macsgt, paramsgt = f(gt, (768,), 'txt')
    macsdh, paramsdh = f(dh, (128,), 'hash')
    macsdf, paramsdf = f(df, (2048,), 'feature')

    total_params = paramsgi + paramsgt + paramsdh + paramsdf
    total_macs = macsgi + macsgt + macsdh * 2 + macsdf * 2

    print('\nTotal stats:')
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', total_macs))
    print('{:<40}  {:<8}'.format('Computational complexity (FLOPs):', total_macs * 2))
    print('{:<40}  {:<8}'.format('Number of parameters:', total_params))


def calculate_stats():
    calculate_stats_for_unhd()
    calculate_stats_for_unhd('thop')


if device.type == 'cpu':
    # test_ptflops()
    # test_thop()
    calculate_stats()
else:
    with torch.cuda.device(device):
        # test_ptflops()
        # test_thop()
        calculate_stats()
