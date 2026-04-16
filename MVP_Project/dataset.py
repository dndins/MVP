import os
import numpy as np
import json
import cv2
from collections import defaultdict
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler

class DataSet(Dataset):
    def __init__(self, json_path, transform, target='path', outside_data_path=None, outside_data_infer=None):
        super(Dataset, self).__init__()
        self.root_path = json_path
        self.transform = transform
        self.target= target
        self.targets = []

        if outside_data_infer is not True: # train valid, test
            # # 数据平均
            self.path_list = json.load(open(json_path, 'r', encoding='utf-8'))[target]
            self.path_list = [x for x in self.path_list if "CLASS1" not in x]
            for x in self.path_list:
                if 'CLASS2' in x:
                    self.targets.append(0)
                elif 'CLASS3' in x:
                    self.targets.append(1)

        else:
            self.folder_list = [os.path.join(outside_data_path, x, "Plane_A") for x in ["CLASS2", "CLASS3"]]
            self.path_list = []
            for folder in self.folder_list:
                img_path_list = [os.path.join(folder, x) for x in os.listdir(folder)]
                for img_A_path in img_path_list:
                    img_B_path = img_A_path.replace("_A", "_B")
                    if os.path.exists(img_B_path):
                        self.path_list.append(img_A_path)
    
            for x in self.path_list:
                if 'CLASS2' in x:
                    self.targets.append(0)
                elif 'CLASS3' in x:
                    self.targets.append(1)

        # 每一个样本的序列
        self.instances_id = [i for i in range(len(self.path_list))]   
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_A_path = self.path_list[idx]
        img_B_path = img_A_path.replace('_A', '_B')

        # 该图像在序列中的位置
        instance_id = self.instances_id[idx]
        # 该图像的标签
        label = self.targets[idx]

        if not os.path.exists(img_A_path):
            print(img_A_path)
        if not os.path.exists(img_B_path):
            print(img_B_path)
        
        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        
        H_A, W_A = img_A.size
        H_B, W_B = img_B.size
        
        img_A = img_A.convert("RGB")
        img_A = self.transform(img_A).float()
        
        img_B = img_B.convert("RGB")
        img_B = self.transform(img_B).float()
        
        label = torch.tensor(label)
        return img_A, img_B, label, instance_id, img_A_path, [H_A, W_A, H_B, W_B]

    def pad_to_square(self, img, fill_color=(0, 0, 0)):

            # 获取图像的宽度和高度
            width, height = img.size
            
            # 计算正方形的边长
            new_size = max(width, height)
            
            # 创建一个新的正方形图像，并将原图粘贴到中心
            padded_img = Image.new("RGB", (new_size, new_size), fill_color)
            padded_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
            
            return padded_img


    def crop_zero_padding(self, img, threshold=10, idx = None, out_size=None):
        """
        先裁剪掉图像的零填充区域，再中心镜像填充到512x512，可选resize到指定尺寸
        img: PIL.Image - 输入图像
        threshold: > threshold 视为有效像素
        out_size: 若不为 None，则最终 resize 到 (out_size, out_size)
        return: PIL.Image - 处理后的图像
        """
        # 1. 将PIL图像转换为numpy数组 (H, W, C)，RGB格式
        img_np = np.array(img)
        original_mode = img.mode  # 保存原始图像模式，用于最终转换
        
        # ---------- 1. 找有效区域并裁剪 ----------
        if img_np.ndim == 3:
            # 对RGB通道判断，只要有一个通道像素值大于阈值就视为有效
            mask = np.any(img_np[..., :3] > threshold, axis=2)
        else:
            # 灰度图直接判断
            mask = img_np > threshold

        coords = np.where(mask)

        if coords[0].size > 0:
            # 计算有效区域的边界
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            # 裁剪有效区域 (PIL的crop格式是 (left, upper, right, lower))
            img_cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            # 转换为numpy数组用于后续填充
            img_cropped_np = np.array(img_cropped)
        else:
            # 如果全是无效像素，保持原图
            img_cropped_np = img_np

        # ---------- 2. 中心镜像填充到512x512 ----------
        # 获取裁剪后图像的尺寸 (H, W)
        h, w = img_cropped_np.shape[:2]
        target_h, target_w = 512, 512
        
        # 计算上下左右需要填充的像素数
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        
        # 镜像填充 (注意：cv2的borderType不影响通道顺序，只是处理边界)
        img_padded_np = cv2.copyMakeBorder(
            img_cropped_np,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REFLECT_101  # 自然的镜像填充模式
        )
        
        # ---------- 3. 可选 resize ----------
        if out_size is not None:
            # 转换为PIL Image进行resize（保持RGB通道顺序）
            img_padded = Image.fromarray(img_padded_np, mode=original_mode)
            img_padded = img_padded.resize((out_size, out_size), Image.BILINEAR)
        else:
            # 直接转换为PIL Image
            img_padded = Image.fromarray(img_padded_np, mode=original_mode)
            # img_padded.save(f'/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/img_path/img{idx}.png')
        return img_padded


class balance_sampler(Sampler):
    def __init__(self, labels, batch_size, num_batches=None):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(set(labels)) 
        assert batch_size % self.num_classes == 0, "Batch size must equal number of classes for 1 sample per class."

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_batches = num_batches if num_batches is not None else 1000  # 设置一个默认的 batch 数目

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                batch.append(random.choice(self.class_to_indices[cls]))  # 随机放回采样
                batch.append(random.choice(self.class_to_indices[cls]))  # 随机放回采样
            yield batch

    def __len__(self):
        return self.num_batches

