import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from einops.layers.torch import Rearrange

######################################## INR(LIIF) ########################################


### 定义解码器隐藏层数（在INR中进行定义）
# hidden_list = [256, 256, 256]   
L = 4  # 位置编码相关

# 获取坐标网格
def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



#TODO 各种简单激活函数
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))
    
#TODO SIREN正弦激活函数
class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=30):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('first_layer_sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def init_weights(m):
    # if hasattr(modules, 'weight'):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # See supplement Sec. 1.5 for discussion of factor 30
        m.weight.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)



#TODO 改进点：MLP解码器
class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_list,
                 act = 'relu',
                 act_trainable = True,
                 **kwargs):
        super().__init__()
        
        # 选择MLP中的激活函数
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        elif act.lower() == 'expsin':
            self.act = ExpSinActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
        elif act.lower() == 'gaussian':
            self.act = GaussianActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
        elif act.lower() == 'quadratic':
            self.act = QuadraticActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
        elif act.lower() == 'multi_quadratic':
            self.act = MultiQuadraticActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
        elif act.lower() == 'laplacian':
            self.act = LaplacianActivation(a=kwargs.get('a', 1.0), trainable=act_trainable) 
        elif act.lower() == 'super_gaussian':
            self.act = SuperGaussianActivation(a=kwargs.get('a', 1.0), b=kwargs['b'], trainable=act_trainable)  
        else:
            assert False, f'activation {act} is not supported'



        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class INR(nn.Module):
    def __init__(self,
                 dim,
                 hidden_list = [256, 256, 256],
                 act = "relu",
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        imnet_in_dim = dim
        imnet_out_dim = dim # 不直接解码图像

        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 + 4 * L  
        if self.cell_decode:
            imnet_in_dim += 2
        
        #TODO 此处解码灰度图，输出通道为1
        ############################### TODO 改进点：这里选用不同的解码器  ###############################
        self.imnet = MLP(imnet_in_dim, imnet_out_dim, hidden_list, act)
        #self.imnet = WireMLP(imnet_in_dim, 256, 3, imnet_out_dim)
        #self.imnet = Wire2dMLP(imnet_in_dim, 256, 3, imnet_out_dim)
        ###############################################################################################

    def query_rgb(self, inp, coord, cell=None):
        feat = inp
        if self.feat_unfold:  # 1，特征展开
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, q, h, w = feat.shape
                q_feat = feat.view(bs, q, -1).permute(0, 2, 1)

                bs, q, h, w = feat_coord.shape
                q_coord = feat_coord.view(bs, q, -1).permute(0, 2, 1)
                
                # 位置编码（这样用位置编码？）
                points_enc = self.positional_encoding(q_coord, L=L)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)  

                # 这里的coord本身就有集成位置编码
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]


                inp = torch.cat([q_feat, rel_coord], dim=-1)  # 2，特征、坐标混合相对位置编码、cell

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]


                    inp = torch.cat([inp, rel_cell], dim=-1)



                bs, q = coord.shape[:2]


                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)  #3，激活解码


                
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)  # 4，最后区域聚合
        if self.local_ensemble:
            t = areas[0];
            areas[0] = areas[3];
            areas[3] = t
            t = areas[1];
            areas[1] = areas[2];
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        bs, q, h, w = feat.shape
        ret = ret.view(bs, h, w, -1).permute(0, 3, 1, 2)
        return ret
    

    #TODO inp对应输入特征图（没有解码器），最终输出的尺度与输入尺度相同（没有利用LIIF作为上/下采样器）
    def forward(self, inp):
        h, w = inp.shape[2], inp.shape[3]
        B = inp.shape[0]
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0).repeat(B, 1, 1)
        coord = coord.unsqueeze(0).repeat(B, 1, 1)

        points_enc = self.positional_encoding(coord, L=L)
        coord = torch.cat([coord, points_enc], dim=-1)  

        return self.query_rgb(inp, coord, cell)

    # 正弦位置编码
    def positional_encoding(self, input, L): 
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32).cuda() * np.pi  
        spectrum = input[..., None] * freq  
        sin, cos = spectrum.sin(), spectrum.cos()  
        input_enc = torch.stack([sin, cos], dim=-2)  
        input_enc = input_enc.view(*shape[:-1], -1)  
        return input_enc