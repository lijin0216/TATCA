import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

from modules import neuron
from spikingjelly.clock_driven import surrogate
from models.spiking_vgg_bn import spiking_vgg11_bn 
from models.spiking_resnet import spiking_resnet18 

from utils.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from utils import augmentation

class SNNGradCAM_DVS:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.target_layer.register_full_backward_hook(self.save_gradient)
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_seq, target_category, T):
        self.activations = []
        self.gradients = []
        self.model.eval()
        self.model.zero_grad()
        
        features_list = []
        for t in range(T):
            x_t = input_seq[t] 
            feat_t = self.model.part1(x_t)
            features_list.append(feat_t)
        
        sequence = torch.stack(features_list, dim=0)
        
        # Check for TATCA module
        if hasattr(self.model, 'tatca_module'):
            attended_sequence = self.model.tatca_module(sequence)
        else:
            attended_sequence = sequence
            
        outputs = []
        for t in range(T):
            out_fr = self.model.part2(attended_sequence[t])
            outputs.append(out_fr)
            
        output_mean = torch.stack(outputs, dim=0).mean(dim=0)
        
        one_hot = torch.zeros_like(output_mean)
        one_hot[0][target_category] = 1
        output_mean.backward(gradient=one_hot)
        
        if len(self.gradients) == 0: return None
            
        avg_grads = torch.stack(self.gradients, dim=0).mean(dim=0)
        avg_acts = torch.stack(self.activations, dim=0).mean(dim=0)
        
        weights = torch.mean(avg_grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * avg_acts, dim=1).squeeze()
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().numpy()

def show_cam_on_dvs(dvs_tensor, mask):
    # 1. Create base image (Accumulate events)
    accumulated_img = dvs_tensor.sum(dim=0).sum(dim=0).cpu().numpy()
    
    # 2. Normalize
    accumulated_img = np.clip(accumulated_img, 0, np.percentile(accumulated_img, 98))
    accumulated_img = accumulated_img - accumulated_img.min()
    accumulated_img = accumulated_img / (accumulated_img.max() + 1e-7)
    
    # 3. Convert to RGB (Gray background)
    img_rgb = np.zeros((accumulated_img.shape[0], accumulated_img.shape[1], 3))
    img_rgb[..., 0] = accumulated_img 
    img_rgb[..., 1] = accumulated_img 
    img_rgb[..., 2] = accumulated_img 

    # 4. Resize Heatmap
    mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # 5. Generate Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] 
    
    # 6. Overlay
    cam_img = 0.5 * heatmap + 0.5 * img_rgb
    cam_img = cam_img / np.max(cam_img)
    
    plt.imshow(cam_img)
    plt.axis('off') 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='DVSCIFAR10', type=str, help='DVSCIFAR10 or dvsgesture')
    parser.add_argument('-data_dir', default='./data', type=str)
    parser.add_argument('-baseline_path', required=True, type=str)
    parser.add_argument('-tatca_path', required=True, type=str)
    parser.add_argument('-out_dir', default='./vis_results_dvs', type=str)
    parser.add_argument('-num_samples', default=10, type=int)
    parser.add_argument('-T', default=10, type=int)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data and Model Class
    print(f"Dataset: {args.dataset}")
    if args.dataset == 'DVSCIFAR10':
        transform_test = transforms.Compose([
            augmentation.ToPILImage(),
            augmentation.Resize(48),
            augmentation.ToTensor(),
        ])
        testset = CIFAR10DVS(args.data_dir, train=False, use_frame=True, frames_num=args.T, split_by='number', normalization=None, transform=transform_test)
        num_classes = 10
        ModelClass = spiking_vgg11_bn
        
    elif args.dataset == 'dvsgesture':
        testset = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        num_classes = 11
        ModelClass = spiking_vgg11_bn
        
    surrogate_function = surrogate.PiecewiseQuadratic()

    # 2. Load Models
    print("Loading Models...")
    net_base = ModelClass(T=args.T, num_classes=num_classes, neuron=neuron.BPTTNeuron, surrogate_function=surrogate_function).to(device)
    net_tatca = ModelClass(T=args.T, num_classes=num_classes, neuron=neuron.BPTTNeuron, surrogate_function=surrogate_function).to(device)
    
    net_base.load_state_dict(torch.load(args.baseline_path, map_location=device)['net'], strict=False)
    net_tatca.load_state_dict(torch.load(args.tatca_path, map_location=device)['net'], strict=False)

    # 3. Hook Target Layer (Last Conv in Part2)
    def find_last_conv(module):
        last_conv = None
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    target_layer_base = find_last_conv(net_base.part2)
    target_layer_tatca = find_last_conv(net_tatca.part2)
    
    print(f"Hooking layer: {target_layer_base}")

    cam_base = SNNGradCAM_DVS(net_base, target_layer_base)
    cam_tatca = SNNGradCAM_DVS(net_tatca, target_layer_tatca)
    
    # 4. Search Loop
    indices = np.random.permutation(len(testset))
    count = 0
    
    print("Searching...")
    for idx in indices:
        data, label = testset[idx] 
        
        # Handle list format from dataset
        if isinstance(data, list):
            data = torch.stack(data, dim=0)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        if not isinstance(data, torch.Tensor):
             data = torch.tensor(data)

        input_seq = data.unsqueeze(1).float().to(device) 
        label = int(label)

        with torch.no_grad():
            # Baseline Prediction
            feats = [net_base.part1(input_seq[t]) for t in range(args.T)]
            seq = torch.stack(feats)
            if hasattr(net_base, 'tatca_module'): seq = net_base.tatca_module(seq)
            out_base = torch.stack([net_base.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_base = out_base.argmax(1).item()
            
            # TATCA Prediction
            feats = [net_tatca.part1(input_seq[t]) for t in range(args.T)]
            seq = torch.stack(feats)
            seq = net_tatca.tatca_module(seq)
            out_tatca = torch.stack([net_tatca.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_tatca = out_tatca.argmax(1).item()

        # Filter: Baseline Wrong AND TATCA Correct
        if pred_base != label and pred_tatca == label:
            print(f"Sample {idx}: GT={label}, Base={pred_base}, TATCA={pred_tatca}")
            
            mask_base = cam_base(input_seq, label, args.T)
            mask_tatca = cam_tatca(input_seq, label, args.T)
            
            if mask_base is None or mask_tatca is None: continue
            
            # Visualization Setup 
            plt.figure(figsize=(3.5, 10)) 
            
            # 1. Original Input 
            plt.subplot(3, 1, 1)
            dvs_tensor = data 
            accumulated = dvs_tensor.sum(dim=0).sum(dim=0).numpy()
            accumulated = np.clip(accumulated, 0, np.percentile(accumulated, 98))
            plt.imshow(accumulated, cmap='gray')
            plt.axis('off')
            
            # 2. Baseline Heatmap
            plt.subplot(3, 1, 2)
            show_cam_on_dvs(dvs_tensor, mask_base)
            
            # 3. TATCA Heatmap
            plt.subplot(3, 1, 3)
            show_cam_on_dvs(dvs_tensor, mask_tatca)
            
            # Remove spacing
            plt.subplots_adjust(hspace=0.02, wspace=0)
            
            save_path = os.path.join(args.out_dir, f"dvs_sample_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved {save_path}")
            
            count += 1
            if count >= args.num_samples:
                break

if __name__ == '__main__':
    main()