from re import sub
import torch 
import numpy as np 
import torch.nn as nn 
import torchvision
import text_model 
import img_model 
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
import math
from cells import *
from router import InteractionModule

class collative_model(nn.Module):
    def __init__(self, opt, texts, word_dim=300, lstm_dim=1024):
        super().__init__()
        self.loss_weight1 =torch.FloatTensor((10.0,)).cuda()

        self.text_model = text_model.get_text_encoder(opt,
            texts_to_build_vocab=texts,
            word_embed_dim=word_dim,
            lstm_hidden_dim=lstm_dim)
        self.img_model = img_model.get_img_encoder(opt)
        self.fusion_add = InteractionModule(opt)

    def compose_fusion(self,query_tensor,wrd,stc,cap_len):
        rgn = query_tensor#bkd
        img = torch.mean(rgn,dim=1)
        query_mod1, query_mod2, paths_l1, paths_l2 = self.fusion_add(rgn,img,wrd,stc,cap_len)
        return query_mod1, query_mod2, paths_l1, paths_l2

    def forward_compose(self,img1, mods, img2):
        query_tensor, target_tensor = self.img_model.forward(img1), self.img_model.forward(img2)
        wrd, stc,cap_len = self.text_model(mods)
        query_add1, query_add2, paths_l1, paths_l2  = self.compose_fusion(query_tensor,wrd,stc,cap_len)#->target_tensor
        return  query_tensor, query_add1, query_add2, target_tensor, stc

    def query_eval(self,img1, mods):
        query_tensor = self.img_model.forward(img1)
        wrd, stc, cap_len = self.text_model(mods)
        query_mod1,query_mod2, paths_l1, paths_l2 = self.compose_fusion(query_tensor,wrd,stc,cap_len)
        query_glo = torch.mean(query_mod2,1)
        return query_glo
        # return query_tensor,query_glo, paths_l1, paths_l2

    def target_eval(self, img2):
        target_tensor = self.img_model.forward(img2)
        target_glo = torch.mean(target_tensor,1)
        return target_glo
        # return target_tensor,target_glo

    def compute_sim(self, query_glo, target_glo):#baseline
        #N.196,1024   #N,1024
        query_glo = F.normalize(query_glo, p=2, dim=-1)
        target_glo = F.normalize(target_glo, p=2, dim=-1)
        sim_all = query_glo @ target_glo.transpose(-1,-2)
        return sim_all

    def compute_loss(self, img1, mods, img2):
        query_tensor, query_add1, query_add2, target_tensor, stc = self.forward_compose(img1, mods, img2)
        query_add1_glo = torch.mean(query_add1,1)
        query_add2_glo = torch.mean(query_add2,1)
        target_glo = torch.mean(target_tensor,1)
        router_glo = torch.cat([query_add1_glo,query_add2_glo],dim=-1)

        sim = self.compute_sim(query_add2_glo, target_glo)
        loss = {}
        loss['class'] = self.compute_batch_based_classification_loss_(sim) #self.ranking(sim)
        loss['sim_MSE'] = self.sim_MSE(router_glo,target_glo)                
        return loss
    
    def compute_batch_based_classification_loss_(self, x):
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight1 * x, labels)   # loss_weight temperature
        return loss

    def sim_MSE(self,query_glo,target_glo):
        query = F.normalize(query_glo, p=2, dim=-1)
        target = F.normalize(target_glo, p=2, dim=-1)
        masks = torch.eye(query.size(0)).cuda()
        A = query@query.permute(1,0)
        B = target@target.permute(1,0)

        loss_kl = torch.nn.MSELoss()(A,B)
        return loss_kl

    def ranking(self, scores):
        margin = 0.2
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        return cost_s.sum() + cost_im.sum()