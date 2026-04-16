# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from dataset import DataSet
# from torchvision import transforms
# from tqdm import tqdm
# import torchvision
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
# from utils import *
# import time
# from train_End2End import *
# import json
# from sklearn.metrics import roc_auc_score

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# class CAMGenerator:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.features = None
#         self.gradients = None
        
#         # 注册钩子
#         target_layer.register_forward_hook(self.save_features)
#         target_layer.register_full_backward_hook(self.save_gradients)
    
#     def save_features(self, module, input, output):
#         self.features = output.detach()
    
#     def save_gradients(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_image, va ,vb, target_class=None):
#         # 前向传播
#         xa, xb, output = self.model(input_image, va ,vb)
        
#         if target_class is None:
#             target_class = np.argmax(output.cpu().data.numpy())
        
#         # 反向传播
#         self.model.zero_grad()
#         one_hot = torch.zeros((1, output.size()[-1]))
#         one_hot[0][target_class] = 1
#         one_hot = one_hot.to(input_image.device)
#         output.backward(gradient=one_hot, retain_graph=True)
        
#         # 计算权重
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
#         # 生成CAM
#         cam = torch.sum(weights * self.features, dim=1, keepdim=True)
#         cam = torch.relu(cam)  # ReLU激活
#         cam = cam - torch.min(cam)
#         cam = cam / torch.max(cam)
        
#         return cam.squeeze().cpu().numpy(), output

# def apply_cam_on_image(img, cam, shape_HW, alpha=0.3):
#     # 调整CAM大小以匹配原始图像
#     img = cv2.resize(img, (shape_HW[0].item(), shape_HW[1].item()))
#     cam = cv2.resize(cam, (shape_HW[0].item(), shape_HW[1].item()))
#     cam = np.uint8(255 * cam)
#     heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
#     # 将原始图像和热图叠加
#     superimposed_img = heatmap * alpha + img * (1 - alpha)
#     superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
#     return superimposed_img

# def infer_with_cam(net, device, json_path, root_path, module, weight_path, save_cam_dir):
#     import os
#     os.makedirs(save_cam_dir, exist_ok=True)
    
#     device = torch.device(device)
#     net.to(device)
    
#     # 获取目标层（ResNet18的最后一个卷积层）
#     target_layer = net.backbone.layer4[-1].conv2
    
#     # 创建CAM生成器
#     cam_generator = CAMGenerator(net, target_layer)
    
#     valid_d = DataSet(json_path, transform=transform, target="test", outside_data_path=opt.outside_data_path, outside_data_infer=opt.outside_data_infer)
#     valid_load = DataLoader(valid_d, batch_size=1, shuffle=True, num_workers=1)
    
#     pred_labels = []
#     pred_list = []
#     label_list = []
#     incorreect_dict = {}
    
#     net.eval()
    
#     pred_dict = {}
#     for i, (img_A, img_B, labels, ids, name, shape_AB) in enumerate(valid_load):
#         img_A, img_B, labels = img_A.to(device), img_B.to(device), labels.to(device)
        
#         va = torch.ones((2, 128)).to(device)
#         vb = torch.ones((2, 128)).to(device)
#         img = torch.cat([img_A, img_B], dim=0)

#         # 生成CAM
#         cam, prob = cam_generator.generate_cam(img, va ,vb)
        
#         # 获取原始图像（反归一化）
#         img_np = img_B.squeeze().cpu().numpy().transpose(1, 2, 0)
#         img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
#         img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
#         # 应用CAM
#         cam_img = apply_cam_on_image(img_np, cam[1], shape_AB[2:])
        
#         # 保存CAM图像
#         filename = os.path.basename(name[0].replace("A", "B")).split('.')[0]
#         save_path = os.path.join(save_cam_dir, f"{filename}_cam.jpg")
#         cv2.imwrite(save_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
        
    
# if __name__ == '__main__':
#     set_seed(42)  # 42是示例种子，可以选择任何整数
#     opt = para()

#     net = NetWork(num_class=opt.num_class)
#     net.load_state_dict(torch.load(opt.test_weight_path))
    
#     save_dir = r'/mnt/data1/zzy/ProjecT/MVP_Project/log/ResNet18_MVP_sigmod/cam_outside'
#     os.makedirs(save_dir, exist_ok=True)

#     infer_with_cam(net, opt.device, opt.json_path, opt.save_path, opt.module, opt.weight_path, save_cam_dir=save_dir)


import os
print(len(os.listdir(r'/mnt/data1/zzy/Data/Norm_Stable_Vulnerable_Crop_outside_V3/CLASS3/Plane_A')))