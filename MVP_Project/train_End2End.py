import os
import torch
import torchvision
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional
from dataset import DataSet, balance_sampler
from torchvision import transforms
from tqdm import tqdm

import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from net_worker import NetWork
from losses import *
from utils import MemoryBank, for_and_backward_block, record_txt, set_seed, save_epoch_curves, linear_alpha
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"✅ Detected {device_count} CUDA device(s).")

strong_transform = create_transform(
            input_size=224,
            is_training=True,
            auto_augment="rand-m5-mstd0.5-inc1",
            interpolation='bicubic',
            re_prob=0,
            re_mode=0,
            re_count=0,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
transform = {'train': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.5, hue=0.5),
                                transforms.ToTensor(),        
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
            'valid': transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
            }

# transform = {'train': strong_transform,
#              'valid': transforms.Compose([
#                                 transforms.Resize((224, 224)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                 ])}

def main(net, epochs, batch_size, lr, device, weight_path, json_path, root_path, module, opt):
    
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    
    device = torch.device(device)
    net.to(device)
    
    # 损失函数
    criterion_ce = nn.BCEWithLogitsLoss()
    criterion_nce = Sup_InfoNCE()

    optimizer = torch.optim.AdamW(net.parameters(), lr)

    lr_step = CosineAnnealingLR(optimizer, T_max=epochs//4, eta_min=lr*0.01)
    
    train_d = DataSet(json_path, transform=transform['train'], target='train')
    labels = [train_d[i][2].item() for i in range(len(train_d))]
    sampler = balance_sampler(labels, batch_size=8, num_batches=500)
    train_load = DataLoader(train_d, batch_sampler=sampler, num_workers=8)
    # train_load = DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=8)

    valid_d = DataSet(json_path, transform=transform['valid'], target='valid')
    valid_load = DataLoader(valid_d, batch_size=batch_size, shuffle=True, num_workers=1)

    # 初始化bank
    bank_A = MemoryBank(size=len(train_d), dim=128, class_num=2, device=device)
    bank_B = MemoryBank(size=len(train_d), dim=128, class_num=2, device=device)
    
    bank_A.get_init_memory_bank(data_loader=train_load, device=device, view='A')
    bank_B.get_init_memory_bank(data_loader=train_load, device=device, view='B')

    
    # 记录训练集的损失 验证集的损失
    loss_train_ce, loss_train_nce, loss_valid_ce, loss_valid_nce = [], [], [], []

    # 记录验证集的精度
    valid_acc_list, valid_auc_list, valid_f1_list = [], [], []
    # 记录训练集每个类的精度 验证集每个类的精度
    valid_per_acc_list = [[], []]
    
    # 记录最好平均精度和整体精度
    best_auc = 0
    best_acc = 0
    best_score = 0
    
    for epoch in tqdm(range(epochs), ncols=100, colour='WHITE'):
        train_loss_ce, train_loss_nce = 0, 0
        valid_loss_ce, valid_loss_nce = 0, 0
        prob_list, pred_list, label_list = [], [], []
        
        for img_A, img_B, labels, ids, _, _ in train_load:
            # print(img_A.shape)
            # 图像A 图像B 图像标签 图像在Memory Bank的位置
            optimizer.zero_grad()
            img_A, img_B, labels, = img_A.to(device), img_B.to(device), labels.to(device)
            img = torch.cat([img_A, img_B], dim=0)
            
            va = torch.stack(bank_A.get_class_mean_feature())
            vb = torch.stack(bank_B.get_class_mean_feature())

            # 获得输出结果
            xa, xb, out = net(img, va, vb)

            # # 分别对两个分支做损失 更新bank
            # loss_nce_A= for_and_backward_block(Memory_bank=bank_A, query=xa, labels=labels, ids=ids, criterion_nce=criterion_nce)
            # loss_nce_B= for_and_backward_block(Memory_bank=bank_B, query=xb, labels=labels, ids=ids, criterion_nce=criterion_nce)
            
            # loss_nce = (loss_nce_A + loss_nce_B)/2
            
            loss_ce = criterion_ce(out.squeeze(1), labels.float())
            
            # lr = linear_alpha(epoch, 100, 1, 0.1)
            # loss = loss_ce + lr*loss_nce
            loss = loss_ce

            loss.backward()
            optimizer.step()

            train_loss_ce += loss_ce.item()
            # train_loss_nce += loss_nce.item()
        
        lr_step.step()
      
        mean_train_loss_ce = train_loss_ce / len(train_load)
        mean_train_loss_nce = train_loss_nce / len(train_load)
        
        loss_train_ce.append(mean_train_loss_ce)
        loss_train_nce.append(mean_train_loss_nce)
        
        with torch.no_grad():
            net.eval()
            for i, (img_A, img_B, labels, ids, _, _) in enumerate(valid_load):
                # 图像A 图像B 图像标签 图像在序列的位置
                img_A, img_B, labels = img_A.to(device), img_B.to(device), labels.to(device)
                # 输入图像 获得预测向量 
                img = torch.cat([img_A, img_B], dim=0)
                
                va = torch.stack(bank_A.get_class_mean_feature())
                vb = torch.stack(bank_B.get_class_mean_feature())
                
                xa, xb, prob = net(img, va, vb)
                
                # 只计算 不更新
                # loss_nce_A = for_and_backward_block(Memory_bank=bank_A, query=xa, labels=labels, ids=ids, criterion_nce=criterion_nce, train=False)
                # loss_nce_B= for_and_backward_block(Memory_bank=bank_B, query=xb, labels=labels, ids=ids, criterion_nce=criterion_nce, train=False)
                
                # loss_nce = (loss_nce_A + loss_nce_B)/2
                
                loss_ce = criterion_ce(prob.squeeze(1), labels.float())
                prob = torch.sigmoid(prob.squeeze(1))
                
                valid_loss_ce += loss_ce.item()
                # valid_loss_nce += loss_nce.item()

                # 保留验证结果
                pred = (prob > 0.5).float()
                prob_list += prob.cpu().tolist()
                pred_list += pred.cpu().tolist()
                label_list += labels.cpu().tolist()
            
            mean_valid_loss_ce = valid_loss_ce / len(valid_load)
            mean_valid_loss_nce = valid_loss_nce / len(valid_load)
            loss_valid_ce.append(mean_valid_loss_ce)
            loss_valid_nce.append(mean_valid_loss_nce)
            
            # 记录验证信息
            tqdm.write(classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False))
            report = classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False)
            accuracy = accuracy_score(label_list, pred_list)
            F1_score = f1_score(label_list, pred_list)
            AUC = roc_auc_score(label_list, prob_list, multi_class='ovr', average='macro')
            cm = confusion_matrix(label_list, pred_list)
            class_acc = cm.diagonal() / cm.sum(axis=1)

            tqdm.write(f"AUC: {AUC:.3f}")
            tqdm.write(f"F1: {F1_score:.3f}")
            for i, acc in enumerate(class_acc):
                tqdm.write(f"Class {i} Accuracy: {acc:.3f}")
                valid_per_acc_list[i].append(acc)
        
            
            valid_acc_list.append(accuracy)
            valid_f1_list.append(F1_score)
            valid_auc_list.append(AUC)
            # 记录整体最高精度
            if np.mean(class_acc)> best_acc:
                best_acc =  np.mean(class_acc)
                torch.save(net.state_dict(), weight_path + '/best_ACC.pth')

            
            tqdm.write(f"epoch: {epoch}       train loss ce: {mean_train_loss_ce:.3f}       train loss nce: {mean_train_loss_nce:.3f}")
            tqdm.write(f"epoch: {epoch}       valid loss ce: {mean_valid_loss_ce:.3f}       valid loss nce: {mean_valid_loss_nce:.3f}")
            tqdm.write(f"best_acc: {best_acc:.3f}")
            tqdm.write(f"best_auc: {best_auc:.3f}")
            
            current_score = (np.mean(class_acc) + AUC) /2

            # 记录平均最高精度 当平均精度最大时 保留记录
            if current_score > best_score:
                best_score = current_score
                torch.save(net.state_dict(), weight_path + '/best.pth')
                record_txt(None, root_path, json_path, module, epochs, batch_size, lr, weight_path, report, class_acc, cm, AUC)
            
            if AUC > best_auc:
                best_auc = AUC
                torch.save(net.state_dict(), weight_path + '/best_AUC.pth')
                record_txt(None, root_path, json_path, module, epochs, batch_size, lr, weight_path, report, class_acc, cm, AUC)


        # 画图
        save_epoch_curves(root_path, \
                          loss_train_ce, loss_train_nce, \
                          loss_valid_ce, loss_valid_nce, \
                          valid_acc_list, valid_f1_list, valid_auc_list, \
                          valid_per_acc_list)

def para():
    arg = argparse.ArgumentParser() 
    arg.add_argument('--module', type=str, default='ResNet18')
    arg.add_argument('--epochs', type=int, default=50)
    arg.add_argument('--batch_size', type=int, default=4)
    arg.add_argument('--lr', type=float, default=1e-4)
    arg.add_argument('--num_class', type=int, default=1)
    arg.add_argument('--device', type=str, default='cuda:0')
    arg.add_argument('--root_path', type=str, default='/mnt/data1/zzy/ProjecT/MVP_Project')
    arg.add_argument('--outside_data_path', type=str, default='/mnt/data1/zzy/Data/Norm_Stable_Vulnerable_Crop_outside_V3')
    arg.add_argument('--json_file', type=str, default='/Norm_Stable_Vulnerable_Crop_add7.json')
    arg.add_argument('--outside_data_infer', type=bool, default=False)
    
    # 测试权重
    arg.add_argument('--test_weight_path', type=str, default= '/mnt/data1/zzy/ProjecT/MVP_Project/log/ResNet18_MVP_sigmod_wo_CL_crop/weight/best.pth')
    opt = arg.parse_args()

    # 保存信息文件夹
    opt.save_path = opt.root_path + "/log/" + opt.module + '_MVP_sigmod_wo_CL_crop'
    os.makedirs(opt.save_path, exist_ok=True)
    opt.weight_path = opt.save_path + '/weight'
    os.makedirs(opt.weight_path, exist_ok=True)
    opt.json_path = opt.root_path + opt.json_file

    return opt

if __name__ == '__main__':
    set_seed(42)
    opt = para()

    if opt.module == 'ResNet18':
       net = NetWork(num_class=opt.num_class)

    main(net, opt.epochs, opt.batch_size, opt.lr, 
         opt.device, opt.weight_path, opt.json_path, opt.save_path, opt.module, opt)