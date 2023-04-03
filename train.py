import os 
import sys
import argparse
import logging
import warnings 
import time 
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
import torchvision
from tqdm import tqdm
import tensorboard_logger as tb_logger

import utils
import datasets
import model as img_text_model
import test
from PIL import Image

# from torch.cuda.amp import autocast as autocast, GradScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(8)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
parser.add_argument('--name', default = 'toptee', help = "data set type")
parser.add_argument('--fashioniq_path', default = '/opt/data/private/kevin/data/fashion-iq/') #replace the path
parser.add_argument('--shoes_path', default = '/opt/data/private/kevin/data/shoes/')

parser.add_argument('--optimizer', default = 'adam')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.0)

parser.add_argument('--learning_rate', type=float, default=1e-4)  
parser.add_argument('--lr_decay', type=int, default=10)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=20)  

parser.add_argument('--class_weight', type=float, default=1.0)  
parser.add_argument('--sim_weight', type=float, default=1.0)  

parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--embed_size', type=int, default=1024)
parser.add_argument('--hid_router', type=int, default=512)
parser.add_argument('--img_encoder', default = 'resnet50')
parser.add_argument('--text_encoder', default= 'LSTM')
parser.add_argument('--grid_num', type=int, default=49)

args = parser.parse_args()


def load_dataset():
    """Loads the input datasets."""
    print('Reading dataset ', args.dataset)
    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path = args.fashioniq_path,
            name = args.name,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize([256,256]),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.FashionIQ(
            path = args.fashioniq_path,
            name = args.name,
            split = 'val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize([256,256]),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))


    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = args.shoes_path,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize([256,256]),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Shoes(
            path = args.shoes_path,
            split = 'test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize([256,256]),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    else:
        print('Invalid dataset', args.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset

def create_model_and_optimizer(opt,texts):
    """Builds the model and related optimizer."""
    print('Creating model and optimizer')
    model = img_text_model.collative_model(opt, texts).cuda()
    optimized_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(optimized_parameters, lr = args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)

    return model, optimizer

def train(model, optimizer, dataloader,testset, epoch, ema):
    """
    train on one epoch
    """

    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    class_loss_avg = utils.RunningAverage()
    sim_MSE_loss = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):
            model.train()

            img1 = data['source_img_data'].cuda()#32,3,224,224
            img2 = data['target_img_data'].cuda()
            mods = data['mod']['str']#32
                
            optimizer.zero_grad()
            loss = model.compute_loss(img1, mods, img2)
            new_loss = args.class_weight * loss['class'] + args.sim_weight * loss['sim_MSE']
            model.Eiters +=1
            model.logger.update('lr', optimizer.param_groups[0]['lr'])
            model.logger.update('Le/loss', new_loss.data.item(), img1.size(0))
            model.logger.update('Le/class', loss['class'].data.item(), img1.size(0))
            model.logger.update('Le/sim_MSE', loss['sim_MSE'].data.item(), img1.size(0))
            model.logger.tb_log(tb_logger, step=model.Eiters)

            new_loss.backward()
            optimizer.step()
            ema.update()

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['new_loss'] = new_loss.item()
                summ.append(summary_batch)
            loss_avg.update(new_loss.item())
            class_loss_avg.update(loss['class'].data.item())
            sim_MSE_loss.update(loss['sim_MSE'].data.item())

            t.set_postfix(loss='{:05.2f}'.format(loss_avg()),
                class_l  ='{:05.3f}'.format(class_loss_avg()),
                sim_MSE  ='{:05.5f}'.format(sim_MSE_loss())
                )
            t.update()
         
def train_and_evaluate(model, optimizer, trainset, testset, model_dir, restore_file=None):

    trainloader = dataloader.DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=args.num_workers)

    best_score = float('-inf')
    ema = utils.EMA(model, 0.999)
    ema.register()
    epoches = args.num_epochs

    for epoch in range(epoches):
        if epoch != 0 and epoch % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div
        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        #train one epoch
        train(model, optimizer, trainloader, testset, epoch, ema)

        tests = []
        ema.apply_shadow()
        for name, dataset in [('test', testset)]:
            t = test.test(args, model, dataset, args.dataset)
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        logging.info(tests)
        for metric_name, metric_value in t:
            tb_logger.log_value(metric_name,metric_value,step=model.Eiters)

        print(tests)
        if args.dataset == 'fashioniq':
            current_score = t[1][1]+t[2][1]
        else:
            current_score = t[0][1]+t[1][1]+t[2][1]
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_json_path_combine = os.path.join(
                model_dir, "metrics_best.json"
            )
            test_metrics = {}
            for metric_name, metric_value in t:
                test_metrics[metric_name] = metric_value
            utils.save_dict_to_json(test_metrics, best_json_path_combine)
            torch.save(model, os.path.join(model_dir, 'train_model.pt'))

if __name__ == '__main__':
    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    
    seed = 300
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(args.model_dir):
        tb_logger.configure(os.path.join(args.model_dir, 'log'), flush_secs=5)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders
    logging.info(args)

    trainset, testset = load_dataset()

    model, optimizer = create_model_and_optimizer(args,[t.encode('utf-8').decode('utf-8') for t in trainset.get_all_texts()])
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    logging.info('- done.')
    train_logger = utils.LogCollector()
    model.Eiters =0
    model.val_iters = 0
    model.loss_temp = 1
    model.logger = train_logger

    # Train the model
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, trainset, testset, args.model_dir, args.restore_file)


