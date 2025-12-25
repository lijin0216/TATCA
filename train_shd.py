import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.clock_driven import functional
from models.spiking_shdnet import create_shd_model

def main():
    parser = argparse.ArgumentParser(description='Train SHD with Multi-Architecture & Attention')
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-b', default=64, type=int, help='Batch size') 
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-T', default=20, type=int, help='Time steps')
    parser.add_argument('-out_dir', default='./logs_shd')
    parser.add_argument('-data_dir', default='./data/shd', help='SHD dataset dir')
    parser.add_argument('-model', default='mlp', type=str, choices=['mlp', 'cnn', 'rnn'], help='Model backbone')
    args = parser.parse_args()

    # Logging setup: logs/shd/model_att
    exp_name = f"{args.model}_{args.att}"
    log_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=== Training SHD ===")
    print(f"Model: {args.model.upper()} ")
    print(f"Saving logs to: {log_dir}")

    try:
        train_set = SpikingHeidelbergDigits(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    except Exception as e:
        print("Warning: Could not load with data_type='frame'. Please check spikingjelly version.")
        raise e

    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=4)

    # Model Initialization
    model = create_shd_model(
        arch_type=args.model,
        input_channels=700, 
        hidden_channels=128, 
        num_classes=20, 
        T=args.T, 
        attn_type=args.att
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        for frame, label in train_loader:
            frame = frame.float().cuda() # (Batch, T, 700)
            label = label.long().cuda()
            
            optimizer.zero_grad()
            
            # 所有模型统一返回 (mean_out, features, attended_features)
            out_mean, _, _ = model(frame)
            
            loss = criterion(out_mean, label)
            loss.backward()
            optimizer.step()
            
            functional.reset_net(model)
            
            train_loss += loss.item() * label.size(0)
            train_acc += (out_mean.argmax(1) == label).float().sum().item()
            train_samples += label.size(0)
            
        # Test Phase
        model.eval()
        test_acc = 0
        test_samples = 0
        
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.float().cuda()
                label = label.long().cuda()
                
                out_mean, _, _ = model(frame)
                
                test_acc += (out_mean.argmax(1) == label).float().sum().item()
                test_samples += label.size(0)
                
                functional.reset_net(model)
        
        acc = test_acc / test_samples * 100
        print(f"Epoch {epoch}: Train Loss {train_loss/train_samples:.4f} | Test Acc {acc:.2f}%")
        
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
            print(f"--> New Best: {max_acc:.2f}%")

if __name__ == '__main__':
    main()