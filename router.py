import torch
import torch.nn as nn
import torch.nn.functional as F
from cells import *
def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class InteractionModule(nn.Module):
    def __init__(self, opt, num_cells=3):
        super(InteractionModule, self).__init__()
        self.opt = opt
        self.num_cells = num_cells
        self.dynamic_itr_l0 = DynamicInteraction_Layer(opt, num_cells, 'layer0')
        self.dynamic_itr_l1 = DynamicInteraction_Layer(opt, num_cells, 'layer1')

    def forward(self, rgn, img, wrd, stc, stc_lens):
        pairs_emb_lst1, paths_l1 = self.dynamic_itr_l0(rgn, img, wrd, stc, stc_lens)
        pairs_emb_lst2, paths_l2 = self.dynamic_itr_l1(pairs_emb_lst1, img, wrd, stc, stc_lens)

        return pairs_emb_lst1, pairs_emb_lst2, paths_l1, paths_l2

class DynamicInteraction_Layer(nn.Module):
    def __init__(self, opt, num_cell, name):
        super(DynamicInteraction_Layer, self).__init__()
        self.opt = opt
        self.name = name
        self.num_cell = num_cell
        self.router = Router(num_cell, opt.embed_size, opt.hid_router)
        self.ric = Rescell(opt)
        self.glo = GlobalCell(opt)
        self.sty = StyleCell(opt)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        sentence_cat = repeat(stc, 'b d -> b k d', k=rgn.shape[1])
        path_in = torch.cat([rgn, sentence_cat], dim=-1)  # N,k,2048,
        path_prob = self.router(path_in)#b，4

        emb_lst = [None] * self.num_cell
        emb_lst[0] = self.ric(rgn)
        emb_lst[1] = self.glo(rgn, img, wrd, stc, stc_lens)
        emb_lst[2] = self.sty(rgn, img, wrd, stc, stc_lens)
        emb_out = torch.stack(emb_lst, dim=1)#b,4,k,d 
        emb = emb_out*unsqueeze2d(path_prob)#b,4,k,d 

        out = emb.sum(dim=1)#b,k,d
        return out,path_prob


