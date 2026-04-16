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
    
    # 初始化memory_bank 并加载初始向量
    def get_init_memory_bank(self, data_loader, device, view):
        # 输入数据加载器  设备 视角类别
        
        with torch.no_grad():
            self.net.eval()
            # 将
            for i, (img_A, img_B, label, inst_id, _, _) in enumerate(data_loader):
                # 图像A 图像B 图像标签 图像在序列的位置
                img_A, img_B, label = img_A.to(device), img_B.to(device), label.to(device)
                
                # memory bank 中向量对应的标签
                self.labels[inst_id] = label.float()
                
                if view == 'A':
                    out = self.net(img_A)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out
                else:
                    out = self.net(img_B)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out

    # 更新bank
    def update(self, indices, features):
        features = torch.nn.functional.normalize(features, dim=1)  # Normalize features
        # Fetch old vectors from the memory bank
        old_features = self.bank[indices]
        # EMA update
        updated_features = self.alpha * old_features + (1 - self.alpha) * features
        updated_features = torch.nn.functional.normalize(updated_features, dim=1)
        # Store updated features back into the memory bank
        self.bank[indices] = updated_features


    # 按照索引寻找对应向量
    def get_features(self, indices):
        return self.bank[indices]
    
    
    # 获得每类的中心向量
    def get_class_mean_feature(self):
        vectors_list = []
        for i in range(self.class_num):
            mean_feature = self.bank[self.labels == i, :].mean(dim=0)
            mean_feature = torch.nn.functional.normalize(mean_feature, dim=-1)
            vectors_list.append(mean_feature)
        return vectors_list

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[2:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        
    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        xx = F.softmax(preds, dim=1)
        xx = torch.clamp(xx, min=1e-4, max=1.0)
        preds_logsoft = torch.log(xx)  # softmax后取对数 [-00, 0]
        

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
    
    # 计算每个类别的聚类中心
    center_vectors = Memory_bank.get_class_mean_feature() # [4, 128]
    matrix_center_vectors = torch.stack(center_vectors).unsqueeze(0).repeat(query.size(0), 1, 1).to(query.device) # [32, 4, 128]
    
    positive_key = matrix_center_vectors[torch.arange(query.size(0)), labels]
    positive_key = F.normalize(positive_key, dim=-1)
    
    negative_keys = []
    
    for class_num in labels:
        neg_idx = Memory_bank.labels != class_num
        negative_key = Memory_bank.get_features(neg_idx)
        negative_keys.append(negative_key)
    
    # 将预测向量 正例中心向量 标签 以及对应的聚类中心向量输入 计算nce loss
    loss = criterion_nce(query, positive_key, negative_keys)  
    if train == True:
    # 用预测向量更新Memory_bank
       Memory_bank.update(ids, query.detach())
    return loss

def linear_alpha(epoch, max_epoch, alpha_start=1.0, alpha_end=0.0):
    """
    线性下降 alpha。

    Args:
        epoch (int): 当前 epoch，从 0 开始
        max_epoch (int): 总 epoch 数
        alpha_start (float): 初始 alpha
        alpha_end (float): 结束 alpha（最后一个 epoch 的 alpha）

    Returns:
        float: 当前 epoch 的 alpha
    """
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

    # 计算 t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)

    color_bar = {0: plt.cm.tab20(0), 1: plt.cm.tab20(2), 2: plt.cm.tab20(14), 3:plt.cm.tab20(8), 4:plt.cm.tab20(1), 5:plt.cm.tab20(3), 6:plt.cm.tab20(5), 7:plt.cm.tab20(7)} # 10


    colors = [color_bar[x] for x in labels]

    # 可视化 t-SNE 结果
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
    """
    自定义 InfoNCE 损失：
    - A1 和 B1 是正样本对。
    - 其他组合 (A1, B2), (A2, B1), (A2, B2) 是负样本对。
    
    Args:
        A (Tensor): 第一个特征向量 (batch_size, 128)。
        B (Tensor): 第二个特征向量 (batch_size, 128)。
        temperature (float): 温度参数 τ。
    Returns:
        Tensor: 损失值。
    """
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    batch_size = A.size(0)
    
    # 切分 A 和 B
    A1, A2 = A[:, :64], A[:, 64:]
    B1, B2 = B[:, :64], B[:, 64:]

    # 计算正样本对相似性
    pos_sim = F.cosine_similarity(A1, B1, dim=-1)  # (batch_size,)
    
    # 计算负样本对相似性
    neg_sim_A1_B2 = torch.matmul(A1, B2.T)  # (batch_size, batch_size)
    neg_sim_A2_B1 = torch.matmul(A2, B1.T)  # (batch_size, batch_size)
    neg_sim_A2_B2 = torch.matmul(A2, B2.T)  # (batch_size, batch_size)

    # 对每个样本构造 logits
    logits = torch.cat([
        pos_sim.unsqueeze(1),  # 正样本相似性 (batch_size, 1)
        neg_sim_A1_B2,
        neg_sim_A2_B1,
        neg_sim_A2_B2
    ], dim=1) / temperature  # 缩放 logits

    # 构造标签：正样本是第一个位置
    labels = torch.zeros(batch_size, dtype=torch.long).to(A.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    return loss

def contrastive_loss(A, B, temperature=0.1):
    """
    计算自定义对比损失：
    A[i, :64] 的正对是 B[i, :64]，负对是 B[j, :64] (j != i)。

    参数:
        A: Tensor [n, 128]，输入向量 A
        B: Tensor [n, 128]，输入向量 B
        temperature: 温度参数

    返回:
        loss: 对比损失
    """
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    
    bs = A.size(0)
    n = A.size(1)

    # 取前 64 维特征
    A_1 = A[:, :n//2]  # [n, 64]
    B_1 = B[:, :n//2]  # [n, 64]
    A_2 = A[:, n//2:]  # [n, 64]
    B_2 = B[:, n//2:]  # [n, 64]

    # 计算余弦相似度矩阵
    logits_up = torch.matmul(A_1, B_1.T)  # [32, 32]
    logits_up /= temperature  # 缩放
    logits_down = torch.matmul(A_1, B_2.T) # [32, 32]
    logits_down /= temperature  # 缩放
    

    # 为负对创建掩码
    mask = torch.eye(bs, dtype=torch.bool).to(A.device)  # [n, n] 对角线为 True
    logits_pos = logits_up[mask].unsqueeze(1)
    logits_neg1 = logits_up[~mask].view(bs, bs-1)
    logits_neg2 = logits_down
    
    logits = torch.cat([logits_pos, logits_neg1, logits_neg2], dim=-1)
    labels = torch.zeros(bs, dtype=torch.long).to(A_1.device)
    # 计算 Softmax 后的 InfoNCE 损失
    loss = F.cross_entropy(logits, labels)

    return loss

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_features(features, method='minmax'):
    """特征归一化"""
    if method == 'standard':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
        print(f"标准化后 - 均值: {np.mean(normalized):.4f}, 标准差: {np.std(normalized):.4f}")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(features)
        print(f"归一化后 - 最小值: {np.min(normalized):.4f}, 最大值: {np.max(normalized):.4f}")
    else:
        normalized = features
        print("未进行归一化")
    return normalized

def plot_tsne_visualization(feature_list, label_list, save_path, perplexity=30, random_state=1):
    """
    绘制t-SNE可视化图，将256维特征分为两个128维特征分别绘制
    
    Args:
        feature_list: 特征列表，每个元素是形状为(1, 256)的张量
        label_list: 标签列表
        save_path: 结果保存路径
        perplexity: t-SNE的困惑度参数
        random_state: 随机种子
    """
    # 转换为numpy数组
    features = np.array([feat[0].cpu().numpy() if hasattr(feat, 'cpu') else feat[0] for feat in feature_list])
    labels = np.array(label_list)
    
    
    # 分离两个128维特征
    feature_a = features[:, 128:]  # 第一个128维特征 (N, 128)
    feature_b = features[:, :128]  # 第二个128维特征 (N, 128)
    # feature_a = normalize_features(features[:, 128:])  # 第一个128维特征 (N, 128)
    # feature_b = normalize_features(features[:, :128])  # 第二个128维特征 (N, 128)
    
    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    
    print("Applying t-SNE to first 128D features...")
    tsne_results_a = tsne.fit_transform(feature_a)
    
    print("Applying t-SNE to second 128D features...")
    tsne_results_b = tsne.fit_transform(feature_b)
    
    # 创建可视化图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 颜色映射
    colors = ['red', 'blue', 'green']
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    # 绘制第一个特征图
    for i in np.unique(labels):  # 假设是二分类
        mask = labels == i
        axes[0].scatter(tsne_results_a[mask, 0], tsne_results_a[mask, 1], 
                       c=colors[i], label=class_names[i], alpha=0.7, s=30)
    
    axes[0].set_title('t-SNE Visualization - First 128D Features', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制第二个特征图
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
    
    # 保存图像
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

    # ===== 4. 综合趋势（可选增强）=====
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

    # ⚠️ 固定文件名 → 自动覆盖
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()