import argparse
import os
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from utils.eeg_loader import get_dataloader
from models.spiking_eegnet import EEG_CNN, EEG_MLP, EEG_RNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-b', default=32, type=int)
    parser.add_argument('-epochs', default=200, type=int)
    parser.add_argument('-T', default=16, type=int, help='SNN simulation steps')
    parser.add_argument('-out_dir', default='./logs_eeg')
    parser.add_argument('-subject', default=1, type=int, help='Subject ID (1-9)')
    args = parser.parse_args()

    subject_dir = os.path.join(args.out_dir, f'S{args.subject}')
    os.makedirs(subject_dir, exist_ok=True)

    train_loader, test_loader, T_signal = get_dataloader(batch_size=args.b, subject_id=args.subject)

    #change model
    model = EEG_CNN(num_classes=4, input_channels=22, T=args.T, signal_length=T_signal).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        for frame, label in train_loader:
            frame = frame.float().cuda() # (N, 22, 63)
            label = label.long().cuda()
            
            optimizer.zero_grad()
            out = model(frame) 
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            functional.reset_net(model)
            
            train_loss += loss.item() * label.size(0)
            train_acc += (out.argmax(1) == label).float().sum().item()
            train_samples += label.size(0)
            
        print(f"Epoch {epoch}: Train Acc {train_acc/train_samples*100:.2f}%")
        
        # Test
        model.eval()
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.float().cuda()
                label = label.long().cuda()
                out = model(frame)
                test_acc += (out.argmax(1) == label).float().sum().item()
                test_samples += label.size(0)
                functional.reset_net(model)
        
        acc = test_acc / test_samples * 100
        print(f"Epoch {epoch}: Test Acc {acc:.2f}%")
        
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), os.path.join(subject_dir, 'best.pth')) 
            print(f"Subject {args.subject} New Best: {max_acc:.2f}%")

if __name__ == '__main__':
    main()