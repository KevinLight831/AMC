import resnet
import torch.nn as nn 
import torch
from transformers import ViTModel,ViTFeatureExtractor

def get_img_encoder(opt):
    if 'res' in opt.img_encoder:
        return resnet_backbone(opt)
    elif 'vit' in opt.img_encoder:
        return vitbackbone(opt)


class resnet_backbone(nn.Module):
    def __init__(self, opt, dim=1024, dropout_p=0.2):
        super().__init__()
        if opt.img_encoder =='resnet50':
            self.resnet = resnet.resnet50(pretrained=True)
        elif opt.img_encoder =='resnet101':
            self.resnet = resnet.resnet101(pretrained=True)
        elif opt.img_encoder =='resnet152':
            self.resnet = resnet.resnet152(pretrained=True)
        elif opt.img_encoder =='resnext101':
            self.resnet = resnet.resnext101_32x8d(pretrained=True)
        else:
            raise ValueError('Unknown backbone source {}'.format(self.backbone_source))

        self.img1x1conv = nn.Conv2d(in_channels=2048, out_channels=opt.embed_size, kernel_size=1,bias=False)
        self.base_feat = 0

        self._init_modules()#resnet

    def _init_modules(self):
        # Build resnet.
        self.base = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu,
                                  self.resnet.maxpool, self.resnet.layer1, self.resnet.layer2, self.resnet.layer3)
        self.top = nn.Sequential(self.resnet.layer4)


    def train(self, mode=True):
        # Override train so that the training mode is set as we want (BN does not update the running stats)
        nn.Module.train(self, mode)
        if mode:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base.apply(set_bn_eval)
            self.top.apply(set_bn_eval)

    def forward(self, im_data):
        base_feat = self.base(im_data)
        top_feat = self.top(base_feat)
        features = self.img1x1conv(top_feat)
        self.base_feat = features
        features = features.flatten(-2,-1).permute(0,2,1)
        return features

class vitbackbone(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.emb_linear = nn.Linear(768,opt.embed_size,bias=False)
        
    def train(self,mode=True):
        nn.Module.train(self,mode)
        if mode:
            def set_ln_eval(m):
                classname = m.__class__.__name__
                if classname.find('LayerNorm') != -1:
                    m.eval()
            self.vit.apply(set_ln_eval)
    
    def forward(self,im_data):
        outputs = self.vit(im_data)
        #outputs.pooler_output b,768
        #print(outputs.last_hidden_state.shape)
        out_features = outputs.last_hidden_state[:,1:,:]#b,197,768 
        features = self.emb_linear(out_features)
        return features 





        

