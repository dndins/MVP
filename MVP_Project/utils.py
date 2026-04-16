import random
from torch import nn
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import numpy as np
from torchvision.models import resnet18
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------Memory Bank-----------------------------------#
class MemoryBank:
    def __init__(self, size, dim, alpha=0.99, class_num=4, device='cuda'):
        self.size = size
        self.dim = dim
        self.alpha = alpha
        self.device = device
        self.class_num = class_num
        self.net = resnet18(pretrained=True)
        input_num = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(input_num, 128))
        self.net.to(device)
        self.bank = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size).to(device)
    
    def get_init_memory_bank(self, data_loader, device, view):
        
        with torch.no_grad():
            self.net.eval()
            # 将
            for i, (img_A, img_B, label, inst_id, _, _) in enumerate(data_loader):
   
                img_A, img_B, label = img_A.to(device), img_B.to(device), label.to(device)
                
                self.labels[inst_id] = label.float()
                
                if view == 'A':
                    out = self.net(img_A)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out
                else:
                    out = self.net(img_B)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out


    def update(self, indices, features):
        features = torch.nn.functional.normalize(features, dim=1)  # Normalize features
        # Fetch old vectors from the memory bank
        old_features = self.bank[indices]
        # EMA update
        updated_features = self.alpha * old_features + (1 - self.alpha) * features
        updated_features = torch.nn.functional.normalize(updated_features, dim=1)
        # Store updated features back into the memory bank
        self.bank[indices] = updated_features

    def get_features(self, indices):
        return self.bank[indices]
    
    def get_class_mean_feature(self):
        vectors_list = []
        for i in range(self.class_num):
            mean_feature = self.bank[self.labels == i, :].mean(dim=0)
            mean_feature = torch.nn.functional.normalize(mean_feature, dim=-1)
            vectors_list.append(mean_feature)
        return vectors_list

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):

        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[2:] += (1-alpha) #  [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        
    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        xx = F.softmax(preds, dim=1)
        xx = torch.clamp(xx, min=1e-4, max=1.0)
        preds_logsoft = torch.log(xx)  
        

        preds_softmax = torch.exp(preds_logsoft) # [0, 1]
        
        
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        
        return loss

def record_txt(incorrect, root_path, json_path, module, epochs, batch_size, lr, weight_path, report, class_acc, cm, auc, outside_infer=None):
    if outside_infer is None:
        with open(root_path + '/best_{}.txt'.format(json_path.split('/')[-1][:-5]), 'w', encoding="utf-8") as f:
            f.write('net: ' + module + '\nepochs: ' + str(epochs) + '\nbatch_size: ' + str(batch_size) + '\nlr: ' +  str(lr)
                    + '\nweight_path: ' + weight_path + '\njson_path: ' + json_path + '\nroot_path: ' + root_path + '\n')
            f.write(report + '\n')
            f.write('best_mean_acc:{}\n'.format(auc))
            for x in range(len(class_acc)):
                f.write('acc{}: {}\n'.format(x, class_acc[x]))
            
            f.write('best_mean_acc:{}\n'.format(class_acc.mean()))
            f.write('confusion_matrix:\n {}\n'.format(cm))
    if outside_infer is not None:
        if outside_infer == False:
            with open(root_path + '/test_{}.txt'.format(json_path.split('/')[-1][:-5]), 'a', encoding="utf-8") as f:
                f.write('\n\n\nInfer: \n')
                f.write(report + '\n')
                f.write('best_mean_acc:{}\n'.format(auc))
                for x in range(len(class_acc)):
                    f.write('acc{}: {}\n'.format(x, class_acc[x]))
                    
                f.write('best_mean_acc:{}\n'.format(class_acc.mean()))
                f.write('confusion_matrix:\n {}\n'.format(cm))
            
                for name, label in incorrect.items():
                    f.write('{}: {}\n'.format(name, label))
        
        if outside_infer == True:
            with open(root_path + '/outside_test_{}.txt'.format(json_path.split('/')[-1][:-5]), 'a', encoding="utf-8") as f:
                f.write('\n\n\nInfer: \n')
                f.write(report + '\n')
                f.write('best_mean_acc:{}\n'.format(auc))
                for x in range(len(class_acc)):
                    f.write('acc{}: {}\n'.format(x, class_acc[x]))
                    
                f.write('best_mean_acc:{}\n'.format(class_acc.mean()))
                f.write('confusion_matrix:\n {}\n'.format(cm))
            
                for name, label in incorrect.items():
                    f.write('{}: {}\n'.format(name, label))
    

def for_and_backward_block(Memory_bank, query, labels, ids, criterion_nce=None, train=True):
    
    query = F.normalize(query, dim=-1)
    
    center_vectors = Memory_bank.get_class_mean_feature() # [4, 128]
    matrix_center_vectors = torch.stack(center_vectors).unsqueeze(0).repeat(query.size(0), 1, 1).to(query.device) # [32, 4, 128]
    
    positive_key = matrix_center_vectors[torch.arange(query.size(0)), labels]
    positive_key = F.normalize(positive_key, dim=-1)
    
    negative_keys = []
    
    for class_num in labels:
        neg_idx = Memory_bank.labels != class_num
        negative_key = Memory_bank.get_features(neg_idx)
        negative_keys.append(negative_key)
    
    loss = criterion_nce(query, positive_key, negative_keys)  
    if train == True:
       Memory_bank.update(ids, query.detach())
    return loss

def linear_alpha(epoch, max_epoch, alpha_start=1.0, alpha_end=0.0):

    if max_epoch <= 1:
        return alpha_end

    alpha = alpha_start + (alpha_end - alpha_start) * (epoch / (max_epoch - 1))
    return float(alpha)


def polt_tsne(feature_list, label_list, save_path):

    label_dict = {0:'RADS1_pred',
                  1:'RADS2_pred',
                  2:'RADS3_pred',
                  3:'RADS4_pred',
                  4:'RADS1_Label',
                  5:'RADS2_Label',
                  6:'RADS3_Label',
                  7:'RADS4_Label',}
    
    features = torch.stack(feature_list).cpu().numpy()
    labels = np.array(label_list)
    save_array = np.concatenate((features, np.expand_dims(labels, axis=1)), axis=1)
    print(save_array.shape)
    np.save('/nvme/ccy/zzy_projects/Carotid_Project/GCN_Layer/train.npy', save_array)
    
    print(features.shape)
    print(labels.shape)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)

    color_bar = {0: plt.cm.tab20(0), 1: plt.cm.tab20(2), 2: plt.cm.tab20(14), 3:plt.cm.tab20(8), 4:plt.cm.tab20(1), 5:plt.cm.tab20(3), 6:plt.cm.tab20(5), 7:plt.cm.tab20(7)} # 10

    colors = [color_bar[x] for x in labels]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=12, c=colors)  

    plt.xticks(range(50, 50, 25))
    plt.yticks(range(50, 50, 25))


    # handles, _ = scatter.legend_elements()
    # legend_labels = [label_dict[i] for i in range(6)]
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bar[i], markersize=10) for i in range(6)]
    # plt.legend(handles, legend_labels, title="Classes")

    # plt.savefig('/mnt/data1/zzy/ProjecT/DYF_US_Synthesis/gen_tsne.png')
    
    # Create custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bar[i], markersize=10) for i in range(len(label_dict))]
    legend_labels = [label_dict[i] for i in range(len(label_dict))]
    plt.legend(handles, legend_labels, title="Classes")
    
    
    # Save the figure
    plt.savefig(save_path + f'/tsne.png', dpi=300)
    

def custom_infonce_loss(A, B, epoch, temperature=0.07):

    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    batch_size = A.size(0)

    A1, A2 = A[:, :64], A[:, 64:]
    B1, B2 = B[:, :64], B[:, 64:]

    pos_sim = F.cosine_similarity(A1, B1, dim=-1)  # (batch_size,)

    neg_sim_A1_B2 = torch.matmul(A1, B2.T)  # (batch_size, batch_size)
    neg_sim_A2_B1 = torch.matmul(A2, B1.T)  # (batch_size, batch_size)
    neg_sim_A2_B2 = torch.matmul(A2, B2.T)  # (batch_size, batch_size)

    logits = torch.cat([
        pos_sim.unsqueeze(1),  
        neg_sim_A1_B2,
        neg_sim_A2_B1,
        neg_sim_A2_B2
    ], dim=1) / temperature  

    labels = torch.zeros(batch_size, dtype=torch.long).to(A.device)

    loss = F.cross_entropy(logits, labels)
    return loss

def contrastive_loss(A, B, temperature=0.1):

    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    
    bs = A.size(0)
    n = A.size(1)

    A_1 = A[:, :n//2]  # [n, 64]
    B_1 = B[:, :n//2]  # [n, 64]
    A_2 = A[:, n//2:]  # [n, 64]
    B_2 = B[:, n//2:]  # [n, 64]

    logits_up = torch.matmul(A_1, B_1.T)  # [32, 32]
    logits_up /= temperature  
    logits_down = torch.matmul(A_1, B_2.T) # [32, 32]
    logits_down /= temperature  

    mask = torch.eye(bs, dtype=torch.bool).to(A.device)  # [n, n] 
    logits_pos = logits_up[mask].unsqueeze(1)
    logits_neg1 = logits_up[~mask].view(bs, bs-1)
    logits_neg2 = logits_down
    
    logits = torch.cat([logits_pos, logits_neg1, logits_neg2], dim=-1)
    labels = torch.zeros(bs, dtype=torch.long).to(A_1.device)

    loss = F.cross_entropy(logits, labels)

    return loss

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_features(features, method='minmax'):

    if method == 'standard':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(features)
    else:
        normalized = features
    return normalized

def plot_tsne_visualization(feature_list, label_list, save_path, perplexity=30, random_state=1):

    features = np.array([feat[0].cpu().numpy() if hasattr(feat, 'cpu') else feat[0] for feat in feature_list])
    labels = np.array(label_list)
    
    

    feature_a = features[:, 128:]  # (N, 128)
    feature_b = features[:, :128]  # (N, 128)
    # feature_a = normalize_features(features[:, 128:])  # (N, 128)
    # feature_b = normalize_features(features[:, :128])  # (N, 128)
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    
    print("Applying t-SNE to first 128D features...")
    tsne_results_a = tsne.fit_transform(feature_a)
    
    print("Applying t-SNE to second 128D features...")
    tsne_results_b = tsne.fit_transform(feature_b)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    colors = ['red', 'blue', 'green']
    class_names = ['Class 0', 'Class 1', 'Class 2']

    for i in np.unique(labels):  
        mask = labels == i
        axes[0].scatter(tsne_results_a[mask, 0], tsne_results_a[mask, 1], 
                       c=colors[i], label=class_names[i], alpha=0.7, s=30)
    
    axes[0].set_title('t-SNE Visualization - First 128D Features', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i in np.unique(labels):
        mask = labels == i
        axes[1].scatter(tsne_results_b[mask, 0], tsne_results_b[mask, 1], 
                       c=colors[i], label=class_names[i], alpha=0.7, s=30)
    
    axes[1].set_title('t-SNE Visualization - Second 128D Features', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(save_path + '/tsne.png', dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved to: {save_path}")
    plt.show()
    
    return tsne_results_a, tsne_results_b



def save_epoch_curves(
    save_dir,
    loss_train_ce, loss_train_nce,
    loss_valid_ce, loss_valid_nce,
    valid_acc_list, valid_f1_list, valid_auc_list,
    valid_per_acc_list
):
    os.makedirs(save_dir, exist_ok=True)

    epochs = len(loss_train_ce)
    x = range(1, epochs + 1)

    plt.figure(figsize=(12, 10))

    # ===== 1. Loss =====
    plt.subplot(2, 2, 1)
    plt.plot(x, loss_train_ce, label='Train CE')
    plt.plot(x, loss_valid_ce, label='Valid CE')
    plt.plot(x, loss_train_nce, label='Train NCE')
    plt.plot(x, loss_valid_nce, label='Valid NCE')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ===== 2. Overall Metrics =====
    plt.subplot(2, 2, 2)
    plt.plot(x, valid_acc_list, label='ACC')
    plt.plot(x, valid_f1_list, label='F1')
    plt.plot(x, valid_auc_list, label='AUC')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # ===== 3. Per-class ACC =====
    plt.subplot(2, 2, 3)
    for i, acc_list in enumerate(valid_per_acc_list):
        if len(acc_list) > 0:
            plt.plot(range(1, len(acc_list) + 1), acc_list, label=f'Class {i}')
    plt.title("Per-Class Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    if len(valid_acc_list) > 0:
        combined = [(a + f + u)/3 for a, f, u in zip(valid_acc_list, valid_f1_list, valid_auc_list)]
        plt.plot(x, combined, label='Mean(ACC,F1,AUC)')
    plt.title("Combined Metric")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()
