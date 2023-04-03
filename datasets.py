
"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import glob
import random
import torchvision


class Shoes(torch.utils.data.Dataset):
    def __init__(self, path, split='train', existed_npy=False, transform=None):
        super(Shoes, self).__init__()
        self.transform = transform
        self.path = path  
        self.readpath = 'relative_captions_shoes.json'
        self.existed_npy = existed_npy
        if split == 'train':
            textfile = 'train_im_names.txt'

        elif split == 'test':
            textfile = 'eval_im_names.txt'

        with open(os.path.join(self.path, self.readpath)) as handle:
            self.dictdump = json.loads(handle.read())
        
        text_file = open(os.path.join(self.path, textfile),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] 
        img_path = os.path.join(self.path,'attributedata')

        self.imgfolder = os.listdir(img_path)
        self.imgfolder = [self.imgfolder[i] for i in range(len(self.imgfolder)) if 'womens' in self.imgfolder[i]]

        ###########################
        if not self.existed_npy:
            self.imgimages_all = []
            for i in range(len(self.imgfolder)):
                path = os.path.join(img_path,self.imgfolder[i])
                imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
                self.imgimages_all += imgfiles
        else:
            self.imgimages_all = np.load(os.path.join(self.path, 'imgimages_all.npy'), allow_pickle=True).tolist()
            
        self.imgs = self.imgimages_all
        self.imgimages_raw = [os.path.basename(imgname) for imgname in self.imgimages_all]
        self.test_targets = []
        self.test_queries = []

        #############################
        if not self.existed_npy:
            self.relative_pairs = self.get_relative_pairs(self.dictdump, imgnames, self.imgimages_all, self.imgimages_raw)
        else:
            if split == 'train':
                self.relative_pairs = np.load(os.path.join(self.path, 'relative_pairs_train.npy'), allow_pickle=True).tolist()
            elif split == 'test':
                self.relative_pairs = np.load(os.path.join(self.path, 'relative_pairs_test.npy'), allow_pickle=True).tolist()

    def get_relative_pairs(self, dictdump, imgnames, imgimages_all, imgimages_raw):
        relative_pairs = []
        for i in range(len(imgnames)):
            ind = [k for k in range(len(dictdump))
                    if dictdump[k]['ImageName'] == imgnames[i]
                    or dictdump[k]['ReferenceImageName'] == imgnames[i]]
            for k in ind:
                if imgnames[i] == dictdump[k]['ImageName']:
                    target_imagename = imgimages_all[imgimages_raw.index(
                        imgnames[i])]
                    source_imagename = imgimages_all[imgimages_raw.index(
                        dictdump[k]['ReferenceImageName'])]
                else:
                    source_imagename = imgimages_all[imgimages_raw.index(
                        imgnames[i])]
                    target_imagename = imgimages_all[imgimages_raw.index(
                        dictdump[k]['ImageName'])]
                text = dictdump[k]['RelativeCaption'].strip()
                relative_pairs.append({
                    'source': source_imagename,
                    'target': target_imagename,
                    'mod': text
                })
        return relative_pairs

    def __len__(self):
        return len(self.relative_pairs)
  
    
    def __getitem__(self, idx):

        caption = self.relative_pairs[idx]
        out = {}
        out['source_img_data'] = self.get_img(caption['source'])
        out['target_img_data'] = self.get_img(caption['target'])
        out['mod'] = {'str': caption['mod']}

        return out

    def get_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def get_img1(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        return img

    def get_all_texts(self):
        if not self.existed_npy:
            text_file = open(os.path.join(self.path, 'train_im_names.txt'),'r')
            imgnames = text_file.readlines()
            imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
            train_relative_pairs = self.get_relative_pairs(self.dictdump, imgnames, self.imgimages_all, self.imgimages_raw)
            texts = []
            for caption in train_relative_pairs:
                mod_texts = caption['mod']
                texts.append(mod_texts)
        else:
            texts = np.load(os.path.join(self.path, 'all_texts.npy'), allow_pickle=True).tolist()
        return texts

    def get_test_queries(self):       # query
        self.test_queries = []
        for idx in range(len(self.relative_pairs)):
            caption = self.relative_pairs[idx]
            mod_str = caption['mod']
            candidate = caption['source']
            target = caption['target']

            out = {}          
            out['source_img_id'] = self.imgimages_all.index(candidate)
            out['source_img_data'] = self.get_img(candidate)
            out['source_img'] = self.get_img1(candidate)
            out['target_img_id'] = self.imgimages_all.index(target)
            out['target_img_data'] = self.get_img(target)
            out['target_img'] = self.get_img1(target)
            out['mod'] = {'str':mod_str}
            self.test_queries.append(out)
        return self.test_queries

    def get_test_targets(self):     
        text_file = open(os.path.join(self.path, 'eval_im_names.txt'),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
        self.test_targets = []
        for i in imgnames:
            out = {}
            out['target_img_id'] = self.imgimages_raw.index(i)
            out['target_img_data'] = self.get_img(self.imgimages_all[self.imgimages_raw.index(i)])
            self.test_targets.append(out)
        return self.test_targets


class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, gallery_all=False, name = 'dress',split = 'train',transform=None):
        super(FashionIQ, self).__init__()

        self.path = path
        self.image_dir = self.path + 'img'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.name = name
        self.split = split
        self.transform = transform
        self.gallery_all = gallery_all

        self.test_targets = []
        self.test_queries = []

        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, self.split)), 'r') as f:
            self.ref_captions = json.load(f)
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(self.name, self.split)), 'r') as f:
            self.images = json.load(f)
    
    def concat_text(self, captions):
        text = "<BOS> {} <AND> {} <EOS>".format(captions[0], captions[1])
        return text
    
    def __len__(self):
        return len(self.ref_captions)
        
    
    def __getitem__(self, idx):
        caption = self.ref_captions[idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir,self.name,image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img
    
    def get_img1(self,image_name):
        img_path = os.path.join(self.image_dir,self.name,image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        return img

    def get_all_texts(self):
        texts = []
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, 'train')), 'r') as f:
            train_captions = json.load(f)
        for caption in train_captions:
            mod_texts = caption['captions']
            texts.append(mod_texts[0])
            texts.append(mod_texts[1])
        return texts

    def get_test_queries(self):       # query
        self.test_queries = []
        for idx in range(len(self.ref_captions)):
            caption = self.ref_captions[idx]
            mod_str = self.concat_text(caption['captions'])
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = self.images.index(candidate)
            out['source_img'] = self.get_img1(candidate)
            out['source_img_data'] = self.get_img(candidate)
            out['target_img_id'] = self.images.index(target)
            out['target_img_data'] = self.get_img(target)
            out['target_img'] = self.get_img1(target)
            out['mod'] = {'str': mod_str}

            self.test_queries.append(out)
        
        return self.test_queries


    def get_test_targets(self):     
        if self.gallery_all:
            self.test_targets = []
            for idx in range(len(self.images)):
                target = self.images[idx]
                out = {}
                out['target_img_id'] = idx
                out['target_img_data'] = self.get_img(target)
                self.test_targets.append(out)
        else:
            test_targets_id = []
            queries = self.get_test_queries()
            for i in queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
        
            self.test_targets = []
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(self.images[i])
                self.test_targets.append(out)   
        return self.test_targets


