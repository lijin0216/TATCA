import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from models import spiking_resnet, spiking_vgg_bn
from modules import neuron

# 1. Custom Dataset for CIFAR-C
class CIFAR_C_Dataset(data.Dataset):
    """
    Custom Dataset class for loading CIFAR-10-C and CIFAR-100-C.
    Data format is expected to be .npy files.
    """
    def __init__(self, data_path, label_path, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.transform = transform
        
        assert len(self.data) == 50000
        assert len(self.labels) == 50000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)
        
        return image, label

# 2. Test Function
def test(net, test_loader, args):
    """
    Independent test function that performs forward propagation and returns accuracy.
    """
    net.eval()
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for i, (frame, label) in enumerate(test_loader):
            frame = frame.float().cuda()
            label = torch.from_numpy(np.array(label)).cuda() 
           
            intermediate_features = []
            for t in range(args.T):
                input_frame_t = frame
                feature_t = net.part1(input_frame_t)
                intermediate_features.append(feature_t)

            sequence = torch.stack(intermediate_features, dim=0)
            attended_sequence = net.tatca_module(sequence)

            total_fr = 0
            for t in range(args.T):
                attended_frame_t = attended_sequence[t]
                out_fr = net.part2(attended_frame_t)
                if t == 0:
                    total_fr = out_fr
                else:
                    total_fr += out_fr
            
            test_samples += label.numel()
            test_acc += (total_fr.argmax(1) == label).float().sum().item()
            
            functional.reset_net(net)
    
    final_acc = test_acc / test_samples
    return final_acc

# 3. Main Execution
def main():
    parser = argparse.ArgumentParser(description='SNN CIFAR-C Test')
    # Key Parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir_c', type=str, required=True, help='Root directory of CIFAR-C dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--model', type=str, default='spiking_resnet18', help='SNN Model architecture name')

    # SNN Parameters
    parser.add_argument('-T', default=6, type=int, help='Simulation time steps')
    parser.add_argument('-tau', default=2, type=float, help='LIF neuron tau')
    parser.add_argument('-b', default=128, type=int, help='Batch size')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='Number of data loading workers')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='Dropout rate')
    
    args = parser.parse_args()
    print(f"--- Starting CIFAR-C Robustness Test ---")
    print(args)

    # Setup Model 
    if args.dataset == 'cifar10':
        num_classes = 10
        c_in = 3
        normalization_mean = (0.4914, 0.4822, 0.4465)
        normalization_std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        num_classes = 100
        c_in = 3
        normalization_mean = (0.5071, 0.4867, 0.4408)
        normalization_std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("Dataset must be 'cifar10' or 'cifar100'")
        
    # Neuron and Surrogate definition
    neuron_model = neuron.BPTTNeuron
    surrogate_function = surrogate_sj.PiecewiseQuadratic() 

    if 'resnet' in args.model:
        net = spiking_resnet.__dict__[args.model](
            neuron=neuron_model, 
            num_classes=num_classes, 
            neuron_dropout=args.drop_rate, 
            tau=args.tau, 
            surrogate_function=surrogate_function, 
            c_in=c_in, 
            T=args.T
        )
    elif 'vgg' in args.model:
        net = spiking_vgg_bn.__dict__[args.model](
            neuron=neuron_model, 
            num_classes=num_classes, 
            neuron_dropout=args.drop_rate, 
            tau=args.tau, 
            surrogate_function=surrogate_function, 
            c_in=c_in, 
            T=args.T
        )
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented.")
    
    net.cuda()

    # Load Weights
    print(f"Loading weights from '{args.checkpoint}'...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    
    # Optional: Print best acc from training if available
    best_acc = checkpoint.get('max_test_acc', 'N/A')
    print(f"Weights loaded successfully! Best test acc in checkpoint: {best_acc}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    
    # Load Labels
    label_path = os.path.join(args.data_dir_c, 'labels.npy')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"File 'labels.npy' not found in '{args.data_dir_c}'")
        
    corruption_files = [f for f in os.listdir(args.data_dir_c) if f.endswith('.npy') and f != 'labels.npy']
    
    all_corruptions_acc = []
    
    print("\n--- Starting Test Loop over Corruptions ---")
    
    for corruption_file in sorted(corruption_files):
        corruption_name = corruption_file.replace('.npy', '')
        data_path = os.path.join(args.data_dir_c, corruption_file)
        
        print(f"\n[Testing Corruption]: {corruption_name}")

        corruption_accs_severity = []
        
        # Iterate over 5 severity levels
        for severity in range(5): 
            start_idx = severity * 10000
            end_idx = (severity + 1) * 10000

            # Create Dataset and Loader
            dataset = CIFAR_C_Dataset(data_path, label_path, transform=transform_test)
            subset = data.Subset(dataset, range(start_idx, end_idx))
            
            test_loader = DataLoader(
                subset, 
                batch_size=args.b, 
                shuffle=False, 
                num_workers=args.j, 
                pin_memory=True
            )
            
            # Run Test
            accuracy_val = test(net, test_loader, args)
            corruption_accs_severity.append(accuracy_val)
            print(f"  Severity {severity+1}: Accuracy = {accuracy_val:.4f}")

        mean_acc_for_corruption = np.mean(corruption_accs_severity)
        all_corruptions_acc.append(mean_acc_for_corruption)
        print(f"-> Mean Accuracy for {corruption_name}: {mean_acc_for_corruption:.4f}")

    final_mean_acc = np.mean(all_corruptions_acc)
    print("\n======================================")
    print(f"mCA (mean Corruption Accuracy) over {len(corruption_files)} corruptions: {final_mean_acc:.4f}")

if __name__ == '__main__':
    main()