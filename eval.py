import os
import sys
import torch
import argparse

import torchvision
from test import test as test_func
import time
import datasets 
from PIL import Image

def load_dataset_eval(args):
    """Loads the input datasets."""
    print('Reading dataset ', args.dataset)
    if args.dataset == 'fashioniq':
        testset = datasets.FashionIQ(
            path = args.data_path,
            name = args.name,
            split = 'val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    elif args.dataset == 'shoes':
        testset = datasets.Shoes(
            path = args.data_path,
            split = 'test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    else:
        print('Invalid dataset', args.dataset)
        sys.exit()
    print('testset size:', len(testset))
    return testset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
    parser.add_argument('--name', default = 'dress', help = "data set type")
    parser.add_argument('--data_path', default = '/opt/data/private/kevin/data/fashion-iq/')
    parser.add_argument('--model_dir',default = './runs/AMC_sim_0/fashioniq/')
    parser.add_argument('--batch_size', type=int, default=32)
    opt = parser.parse_args()
    model_path = os.path.join(opt.model_dir,'train_model.pt')
    model = torch.load(model_path)

    testset = load_dataset_eval(opt)
    print('-----------------------'+opt.model_dir+'-----------'+opt.name+'-----------------')
    t = test_func(opt, model, testset, opt.dataset)
    tests = [('test' + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
    print(tests)

if __name__=='__main__':
    main()
