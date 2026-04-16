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


        self.instances_id = [i for i in range(len(self.path_list))]   
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_A_path = self.path_list[idx]
        img_B_path = img_A_path.replace('_A', '_B')


        instance_id = self.instances_id[idx]

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

            width, height = img.size
            
            new_size = max(width, height)
            
            padded_img = Image.new("RGB", (new_size, new_size), fill_color)
            padded_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
            
            return padded_img


    def crop_zero_padding(self, img, threshold=10, idx = None, out_size=None):

        img_np = np.array(img)
        original_mode = img.mode 
        
        if img_np.ndim == 3:
            mask = np.any(img_np[..., :3] > threshold, axis=2)
        else:
            mask = img_np > threshold

        coords = np.where(mask)

        if coords[0].size > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            img_cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            img_cropped_np = np.array(img_cropped)
        else:
            img_cropped_np = img_np

        h, w = img_cropped_np.shape[:2]
        target_h, target_w = 512, 512
        
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        
        img_padded_np = cv2.copyMakeBorder(
            img_cropped_np,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REFLECT_101 
        )
        
        if out_size is not None:
            img_padded = Image.fromarray(img_padded_np, mode=original_mode)
            img_padded = img_padded.resize((out_size, out_size), Image.BILINEAR)
        else:
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
        self.num_batches = num_batches if num_batches is not None else 1000 

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                batch.append(random.choice(self.class_to_indices[cls])) 
                batch.append(random.choice(self.class_to_indices[cls]))  
            yield batch

    def __len__(self):
        return self.num_batches

