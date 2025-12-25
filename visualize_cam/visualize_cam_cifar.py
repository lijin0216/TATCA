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

from models.spiking_resnet import spiking_resnet18
from models.spiking_vgg_bn import spiking_vgg11_bn 

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

    def __call__(self, input_tensor, target_category, T=6):
        self.activations = []
        self.gradients = []
        self.model.eval()
        self.model.zero_grad()
        
        features_list = []
        for t in range(T):
            x_t = input_tensor 
            feat_t = self.model.part1(x_t)
            features_list.append(feat_t)
        
        sequence = torch.stack(features_list, dim=0) 
        
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
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
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
    parser.add_argument('-data_dir', default='./data', type=str, help='Directory for datasets')
    parser.add_argument('-baseline_path', default='./logs/baseline.pth', type=str)
    parser.add_argument('-tatca_path', default='./logs/tatca.pth', type=str)
    parser.add_argument('-out_dir', default='./vis_results', type=str)
    parser.add_argument('-num_samples', default=5, type=int)
    parser.add_argument('-T', default=6, type=int, help='time steps')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    classes = testset.classes 
    indices = np.random.permutation(len(testset))
    
    surrogate_function = surrogate.PiecewiseQuadratic()
    
    print("Loading Baseline Model...")
    net_base = spiking_resnet18(
        T=args.T, 
        num_classes=100, 
        neuron=neuron.BPTTNeuron,           
        surrogate_function=surrogate_function 
    ).to(device)
    
    checkpoint_base = torch.load(args.baseline_path, map_location=device)
    net_base.load_state_dict(checkpoint_base['net'], strict=False)
    
    print("Loading TATCA Model...")
    net_tatca = spiking_resnet18(
        T=args.T, 
        num_classes=100,
        neuron=neuron.BPTTNeuron,          
        surrogate_function=surrogate_function 
    ).to(device)
    
    checkpoint_tatca = torch.load(args.tatca_path, map_location=device)
    net_tatca.load_state_dict(checkpoint_tatca['net'], strict=False)
    
    target_layer_base = net_base.part2[1][-1].conv2
    target_layer_tatca = net_tatca.part2[1][-1].conv2
    
    cam_base = SNNGradCAM(net_base, target_layer_base)
    cam_tatca = SNNGradCAM(net_tatca, target_layer_tatca)
    
    count = 0
    print(f"Start searching for contrastive samples on CIFAR...")
    
    for idx in indices:
        img_tensor, label_idx = testset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Baseline
            feats = [net_base.part1(img_input) for _ in range(args.T)]
            seq = torch.stack(feats)
            if hasattr(net_base, 'tatca_module'): seq = net_base.tatca_module(seq)
            out_base = torch.stack([net_base.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_base = out_base.argmax(1).item()
            
            # TATCA
            feats = [net_tatca.part1(img_input) for _ in range(args.T)]
            seq = torch.stack(feats)
            seq = net_tatca.tatca_module(seq)
            out_tatca = torch.stack([net_tatca.part2(seq[t]) for t in range(args.T)]).mean(0)
            pred_tatca = out_tatca.argmax(1).item()
            
        if pred_base != label_idx and pred_tatca == label_idx:
            print(f"Sample {idx}: True={classes[label_idx]}, BaseWrong={classes[pred_base]}, TatcaRight={classes[pred_tatca]}")
            
            mask_base = cam_base(img_input, target_category=label_idx)
            mask_tatca = cam_tatca(img_input, target_category=label_idx)
            
            if mask_base is None or mask_tatca is None: continue


            plt.figure(figsize=(3.5, 10))  
            
            # 1. Original image
            plt.subplot(3, 1, 1)
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
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
            

            save_path = os.path.join(args.out_dir, f"cifar_sample_{idx}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved to {save_path}")
            
            count += 1
            if count >= args.num_samples:
                break
                
    print("Done!")

if __name__ == '__main__':
    main()
