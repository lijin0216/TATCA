import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

from modules import neuron
from spikingjelly.clock_driven import surrogate
from models.spiking_vgg_bn import spiking_vgg13_bn 

try:
    from utils.data_loaders import TinyImageNet
except ImportError:
    TinyImageNet = None

class SNNGradCAM:
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

    def __call__(self, input_tensor, target_category, T):
        self.activations = []
        self.gradients = []
        self.model.eval()
        self.model.zero_grad()
        
        # SNN Forward Pass simulation
        features_list = []
        for t in range(T):
            x_t = input_tensor 
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
        
        if len(self.gradients) == 0 or len(self.activations) == 0:
            return None 
            
        avg_grads = torch.stack(self.gradients, dim=0).mean(dim=0)
        avg_acts = torch.stack(self.activations, dim=0).mean(dim=0)
        
        weights = torch.mean(avg_grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * avg_acts, dim=1).squeeze()
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().numpy()

def show_cam_on_image(img_tensor, mask):
    # Standard ImageNet Mean/Std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    inv_normalize = transforms.Normalize(
        mean= -mean / std,
        std= 1 / std
    )
    
    img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] 
    
    cam_img = 0.4 * heatmap + 0.6 * img
    cam_img = cam_img / np.max(cam_img)
    
    plt.imshow(cam_img)
    plt.axis('off')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='./data', type=str, help='Tiny ImageNet dataset directory')
    parser.add_argument('-baseline_path', required=True, type=str)
    parser.add_argument('-tatca_path', required=True, type=str)
    parser.add_argument('-out_dir', default='./vis_results_tiny', type=str)
    parser.add_argument('-num_samples', default=5, type=int)
    parser.add_argument('-T', default=4, type=int, help='Time steps') 
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing (Test set)
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    print(f"Loading Tiny-ImageNet from {args.data_dir}...")
    testset = None
    
    # Try 1: Custom Loader
    if TinyImageNet is not None:
        try:
            print("Trying custom TinyImageNet loader...")
            testset = TinyImageNet(root=args.data_dir, train=False, transform=transform_test)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Custom loader failed ({e}). Trying ImageFolder fallback...")
    
    # Try 2: ImageFolder fallback
    if testset is None:
        valdir = os.path.join(args.data_dir, 'val')
        if not os.path.exists(valdir):
             valdir = os.path.join(args.data_dir, 'tiny-imagenet-200', 'val')
        
        print(f"Trying ImageFolder on {valdir}...")
        try:
            testset = torchvision.datasets.ImageFolder(valdir, transform=transform_test)
        except Exception as e:
            print(f"FATAL ERROR: Could not load dataset. Please check your path.")
            print(f"Expected structure: {args.data_dir}/val/images OR {args.data_dir}/tiny-imagenet-200/val/images")
            raise e

    print(f"Dataset loaded! Size: {len(testset)}")
    indices = np.random.permutation(len(testset))
    
    surrogate_function = surrogate.PiecewiseQuadratic()
    
    # 1. Load Models (VGG13, num_classes=200, c_in=3)
    print("Loading Baseline Model (VGG13)...")
    net_base = spiking_vgg13_bn(
        T=args.T, 
        num_classes=200, 
        neuron=neuron.BPTTNeuron, 
        surrogate_function=surrogate_function,
        c_in=3 
    ).to(device)
    
    net_base.load_state_dict(torch.load(args.baseline_path, map_location=device)['net'], strict=False)
    
    print("Loading TATCA Model (VGG13)...")
    net_tatca = spiking_vgg13_bn(
        T=args.T, 
        num_classes=200,
        neuron=neuron.BPTTNeuron,
        surrogate_function=surrogate_function,
        c_in=3 
    ).to(device)
    
    net_tatca.load_state_dict(torch.load(args.tatca_path, map_location=device)['net'], strict=False)

    # 2. Hook last conv layer
    def find_last_conv(module):
        last_conv = None
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    target_layer_base = find_last_conv(net_base.part2)
    target_layer_tatca = find_last_conv(net_tatca.part2)
    
    print(f"Hooking layer: {target_layer_base}")
    
    cam_base = SNNGradCAM(net_base, target_layer_base)
    cam_tatca = SNNGradCAM(net_tatca, target_layer_tatca)
    
    count = 0
    print(f"Start searching for contrastive samples on Tiny-ImageNet...")
    
    for idx in indices:
        img_tensor, label_idx = testset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Baseline Prediction
            feats = [net_base.part1(img_input) for _ in range(args.T)]
            seq = torch.stack(feats)
            if hasattr(net_base, 'tatca_module'): seq = net_base.tatca_module(seq)
            out_base = torch.stack([net_base.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_base = out_base.argmax(1).item()
            
            # TATCA Prediction
            feats = [net_tatca.part1(img_input) for _ in range(args.T)]
            seq = torch.stack(feats)
            seq = net_tatca.tatca_module(seq)
            out_tatca = torch.stack([net_tatca.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_tatca = out_tatca.argmax(1).item()
            
        # Filter: Baseline Wrong AND TATCA Correct
        if pred_base != label_idx and pred_tatca == label_idx:
            print(f"Sample {idx}: GT={label_idx}, Base={pred_base}, TATCA={pred_tatca}")
            
            mask_base = cam_base(img_input, label_idx, args.T)
            mask_tatca = cam_tatca(img_input, label_idx, args.T)
            
            if mask_base is None or mask_tatca is None: continue

            # Visualization Setup 
            plt.figure(figsize=(3.5, 10)) 
            
            # 1. Original Image
            plt.subplot(3, 1, 1)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inv_normalize = transforms.Normalize(mean= -mean / std, std= 1 / std)
            
            img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
            img_show = np.clip(img_show, 0, 1)
            plt.imshow(img_show)
            plt.axis('off')
            
            # 2. Baseline Heatmap
            plt.subplot(3, 1, 2)
            show_cam_on_image(img_tensor, mask_base)
            
            # 3. TATCA Heatmap
            plt.subplot(3, 1, 3)
            show_cam_on_image(img_tensor, mask_tatca)
            
            # Remove spacing
            plt.subplots_adjust(hspace=0.02, wspace=0)
            
            save_path = os.path.join(args.out_dir, f"tiny_sample_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved to {save_path}")
            
            count += 1
            if count >= args.num_samples:
                break
                
    print("Done!")

if __name__ == '__main__':
    main()