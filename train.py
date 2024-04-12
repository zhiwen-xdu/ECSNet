"""
Author: CHEN ZHIWEN
Date: Jan 2021
"""
from data_utils.NCalDataLoader import NCalDataLoader    # For Object Classifiction
from data_utils.UCFDataLoader import UCFDataLoader      # For Action Classifiction

import argparse
import numpy as np
import os
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))     


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ECSNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='ncal_dot', help='model name [default: cifar10_lstm]')
    parser.add_argument('--num_class', default=101, help='the number of class')
    parser.add_argument('--epoch',  default=120, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default="ncal_101", help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    return parser.parse_args()


def test(model, loader, num_class=11):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('ECSNet_For_NCal')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)


    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)                        


    log_string('Load Dataset ...')
    DATA_PATH = "./data/NCaltech101/"
    # Dataset
    train_dataset = NCalDataLoader(root=DATA_PATH,split='train')
    test_dataset = NCalDataLoader(root=DATA_PATH,split='test')
    log_string('Load Dataloader ...')
    # DataLoader
    train_dataloader  = torch.utils.data.DataLoader(train_dataset , batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size=args.batch_size, shuffle=True,drop_last=True)


    log_string('Load Model ...')
    MODEL = importlib.import_module(args.model)                       
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))   
    shutil.copy('models/util.py', str(experiment_dir))

    classifier = MODEL.get_model(args.num_class)                      
    criterion = MODEL.get_loss().cuda()                               
    classifier = torch.nn.DataParallel(classifier).cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])   
        log_string('Load Pretrain Model')
    except:
        log_string('No Existing Model, Starting Training From Scratch...')   
        start_epoch = 0

 
    log_string('Set Optimizer ...')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    log_string('Set scheduler ...')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0


    logger.info('Start Training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        mean_correct = []
        train_loss = []
        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            features, target = data            
            target = target[:, 0]
            features, target = features.cuda(), target.cuda()        

            optimizer.zero_grad()                                
            classifier = classifier.train()
            pred = classifier(features)                          
            loss = criterion(pred, target.long())             
            train_loss.append(loss.item())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(features.size()[0]))    
            loss.backward()                                      
            optimizer.step()

        global_step += 1
        train_instance_loss = np.mean(train_loss)
        log_string('Train Loss: %f' % train_instance_loss)
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(),test_dataloader,args.num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)


