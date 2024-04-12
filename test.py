from data_utils.UCFDataLoader import UCFDataLoader
import numpy as np
import os
import torch
from tqdm import tqdm
import sys
import importlib
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Eight Traffic
DATA_PATH = "./data/UCF101/"
test_dataset = UCFDataLoader(root=DATA_PATH,split='test')
test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size=2, shuffle=True,drop_last=True)


num_class = 101
MODEL = importlib.import_module("action_classifier")                   
classifier = MODEL.get_model(num_class)                                 
classifier = torch.nn.DataParallel(classifier).cuda()
checkpoint = torch.load("./log/ECSNet_For_UCF101/checkpoints/best_model.pth")
classifier.load_state_dict(checkpoint['model_state_dict'])      


def inference(model, loader, num_class=11):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        events, target = data                           
        target = target[:, 0]                      
        events, target = events.cuda(), target.cuda()    
        classifier = model.eval()                        
        pred = classifier(events)                     
        pred_choice = pred.data.max(1)[1]                
        for cat in np.unique(target.cpu()):              
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(events[target==cat].size()[0])              
            class_acc[cat,1]+=1                                                                  
        correct = pred_choice.eq(target.long().data).cpu().sum()                                 
        mean_correct.append(correct.item()/float(events.size()[0]))

    diff_class_acc = class_acc[:,0]/ class_acc[:,1]                                                         
    class_acc = np.mean(diff_class_acc)
    instance_acc = np.mean(mean_correct)
    return instance_acc,diff_class_acc, class_acc


def save_txt(acc_dict):
    acc_txt = "./log/ECSNet_For_UCF101/performance.txt"
    f1 = open(acc_txt, 'a')
    for key in acc_dict:
        f1.write(key+":"+str(acc_dict[key])+"\n")
    f1.close()


with torch.no_grad():
    cls_flie = os.path.join("./data/UCF101/", 'class_names.txt')
    clses = [line.rstrip() for line in open(cls_flie)]
    start_time = time.time()
    instance_acc, diff_class_acc, class_acc = inference(classifier, test_dataloader, num_class=num_class)

    acc_dict = dict(zip(clses,diff_class_acc))
    save_txt(acc_dict)
    end_time = time.time()
    print(acc_dict,class_acc,instance_acc)
    print("Totally Time Cost:",(end_time-start_time),"s")
