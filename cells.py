import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from einops import rearrange, reduce, repeat
# from block import fusions
from torch.autograd import Variable
import copy
 
def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size*2, hid,bias=False),
                                nn.LayerNorm(normalized_shape = hid),
                                nn.ReLU(True), 
                                nn.Linear(hid, num_out_path,bias=False))

    def forward(self, x):
        x = x.mean(-2)#b,k,d
        x = self.mlp(x)
        soft_g = torch.sigmoid(x)
        return soft_g

class Ensemble(nn.Module):
    def __init__(self, opt):
        super(Ensemble, self).__init__()
        self.opt = opt
        self.ric = Rescell(opt)
        self.glo = GlobalCell(opt)
        self.sty = StyleCell(opt)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        pairs_emb_lst = self.ric(rgn) +self.glo(rgn, img, wrd, stc, stc_lens)+self.sty(rgn, img, wrd, stc, stc_lens)
        return pairs_emb_lst

class Rescell(nn.Module):#NIN CELL
    def __init__(self, opt):
        super(Rescell, self).__init__()
        self.norm1 = nn.LayerNorm(opt.embed_size,elementwise_affine=False)

    def forward(self, rgn):
        emb = self.norm1(rgn)
        return emb


class StyleCell(nn.Module):#GTN CELL
    def __init__(self, opt):
        super(StyleCell, self).__init__()
        self.opt = opt
        self.norm1 = nn.LayerNorm(opt.embed_size,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(opt.embed_size,elementwise_affine=False)
        self.fc_gamma = nn.Sequential(nn.Linear(opt.embed_size, opt.embed_size),)
        self.fc_beta = nn.Sequential(nn.Linear(opt.embed_size, opt.embed_size),)
        self.softmax = nn.Softmax(dim=-1)
   
    def forward(self, rgn, img, wrd, stc, stc_lens):
        img_vector = rgn
        gammas = self.fc_gamma(stc).unsqueeze(-2)
        betas = self.fc_beta(stc).unsqueeze(-2)
        normalized = img_vector * (gammas) + betas
        normalized = self.norm1(normalized)
        return normalized


class GlobalCell(nn.Module):#CRN CELL
    def __init__(self, opt):
        super(GlobalCell, self).__init__()
        self.opt = opt
        self.norm1 =  nn.LayerNorm(opt.embed_size,elementwise_affine=False)
        self.linear = nn.Linear(opt.embed_size*2, opt.embed_size)
        self.SA = SelfAttentionCell(opt)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        sentence_cat = repeat(stc, 'b d -> b k d', k=rgn.shape[1])
        x = torch.cat([rgn, sentence_cat], dim=-1)  # N,k,2048,
        x_out = self.linear(x)
        out = self.SA(x_out)
        out = self.norm1(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, h, is_share=False, drop=0.0):
        super(AttentionLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            self.linears = clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)
        
    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden, drop=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class SelfAttentionCell(nn.Module):
    def __init__(self, opt):
        super(SelfAttentionCell, self).__init__()
        self.h = 8
        self.drop=0.0
        self.mlp_ratio = 0.5
        mlp_hidden_dim = int(opt.embed_size * self.mlp_ratio)
        self.att_layer = AttentionLayer(opt.embed_size, self.h, drop=self.drop)
        self.feed_forward_layer = FeedForward(opt.embed_size, mlp_hidden_dim, drop=self.drop)
        self.dropout = nn.Dropout(self.drop)
        self.norm1 = nn.LayerNorm(opt.embed_size)
        self.norm2 = nn.LayerNorm(opt.embed_size)

    def forward(self, local_emb):
        mask=None 
        self_att_emb = self.dropout(self.att_layer(self.norm1(local_emb), mask=mask))
        out = self_att_emb + self.dropout(self.feed_forward_layer(self.norm2(self_att_emb)))
        return out
