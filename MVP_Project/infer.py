import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DataSet
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    recall_score, precision_score, 
)
from utils import *
from collections import Counter

from train_End2End import *

transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    eps = 1e-8

    sensitivity = tp / (tp + fn + eps)   # Recall
    specificity = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)           # Precision
    npv = tn / (tn + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    metrics = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV (Precision)': ppv,
        'NPV': npv,
        'F1 Score': f1,
    }

    return metrics


def infer(net, device, json_path, root_path, module, weight_path, opt):
    
    device = torch.device(device)
    net.to(device)
    
    valid_d = DataSet(json_path, transform=transform, target="test", outside_data_path=opt.outside_data_path, outside_data_infer=opt.outside_data_infer)
    valid_load = DataLoader(valid_d, batch_size=1, shuffle=True, num_workers=1)
    
    torch.cuda.synchronize()  

    pred_list = []
    feature_list = []
    prob_list = []
    label_list = []
    csv_data = []
    incorreect_dict = {}
    with torch.no_grad():
        net.eval()
        
        pred_dict = {}

        for i, (img_A, img_B, labels, ids, name, _) in enumerate(valid_load):
            img_A, img_B, labels = img_A.to(device), img_B.to(device), labels.to(device)
            img = torch.cat([img_A, img_B], dim=0)

            va = torch.ones((2, 128)).to(device)
            vb = torch.ones((2, 128)).to(device)

            xa, xb, prob = net(img, va ,vb)
            mid = torch.cat([xa, xb], dim=-1)

            prob = torch.sigmoid(prob.squeeze(1))
            pred = (prob > 0.5).float()

            if pred != labels:
                incorreect_dict[name[0]] = [
                    int(pred + 1),
                    [round(num, 3) for num in prob.tolist()]
                ]

            feature_list.append(mid)

            label_list += labels.cpu().tolist()
            pred_list += pred.cpu().tolist()

            prob_list.append(prob.cpu().detach().numpy())

            pred_dict[name[0].split('/')[-1]] = {"gt":int(labels[0].cpu().numpy()),
                                  "pred":int(pred.cpu().numpy()),
                                  "score":[float(prob[0].cpu().numpy())]}

            for j in range(len(name)):
                csv_data.append({
                    "Name": name[j].split('/')[-1],
                    "GT": int(labels[0].cpu().numpy()),
                    "Pred": int(pred.cpu().numpy()),
                    "Prob": float(prob.cpu().numpy()[j])
                })
        if opt.outside_data_infer is True:
            csv_path = os.path.join(opt.save_path, "test_results_{}.csv".format(opt.outside_data_path.split("/")[-1]))
        else:
            csv_path = os.path.join(opt.save_path, "test_results_inside.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")

        plot_tsne_visualization(feature_list, label_list, root_path)

        auc = roc_auc_score(label_list, prob_list, multi_class='ovr', average='macro')
        print("AUC:", auc)

        bacc = balanced_accuracy_score(label_list, pred_list)
        print("Bacc:", bacc)

        f1 = f1_score(label_list, pred_list, average='macro')
        print("f1-score:", f1)
        
        metrics = calculate_metrics(label_list, pred_list)

        for metric, value in metrics.items():
            if metric in ['TP', 'TN', 'FP', 'FN']:
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
        print(confusion_matrix(label_list, pred_list))

        tqdm.write(classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False))
        report = classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False)
        
        cm = confusion_matrix(label_list, pred_list)
        class_acc = cm.diagonal() / cm.sum(axis=1)
    
    
        record_txt(incorreect_dict, root_path, json_path, module, epochs=None, batch_size=None, lr=None, \
                   weight_path=None, report=report, class_acc=class_acc, cm=cm, auc=auc, outside_infer=opt.outside_data_infer)
        
    
if __name__ == '__main__':
    set_seed(42)  
    opt = para()

    net = NetWork(num_class=opt.num_class)
    net.load_state_dict(torch.load(opt.test_weight_path))
    
    infer(net, opt.device, opt.json_path, opt.save_path, opt.module, opt.weight_path, opt)

