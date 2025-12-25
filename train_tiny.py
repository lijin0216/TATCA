import datetime
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import amp 
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
import random
import numpy as np
import sys
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment

# Local imports
from models import spiking_vgg_bn 
from modules import neuron
from modules import surrogate as surrogate_self
from utils import Bar, AverageMeter, accuracy
from utils.data_loaders import TinyImageNet

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='SNN training with TATCA on Tiny ImageNet')
    
    # Training Configuration
    parser.add_argument('-seed', default=2025, type=int, help='Random seed')
    parser.add_argument('-name', default='TATCA_TinyImageNet', type=str, help='Name for checkpoint and logs')
    parser.add_argument('-T', default=6, type=int, help='Simulation time steps')
    parser.add_argument('-tau', default=2.0, type=float, help='LIF neuron time constant')
    parser.add_argument('-b', default=128, type=int, help='Batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N', help='Total training epochs')
    parser.add_argument('-j', default=10, type=int, metavar='N', help='Number of data loading workers')
    
    # Dataset and Output
    parser.add_argument('-data_dir', type=str, default='./data', help='Directory for Tiny ImageNet dataset')
    parser.add_argument('-dataset', default='tiny-imagenet', type=str, help='Dataset name (fixed)')
    parser.add_argument('-out_dir', type=str, default='./logs', help='Root directory for logs and checkpoints')
    
    # Model and Optimization
    parser.add_argument('-model', type=str, default='spiking_vgg13_bn', help='SNN model architecture (e.g., spiking_vgg13_bn)')
    parser.add_argument('-surrogate', default='triangle', type=str, help='Surrogate gradient function')
    parser.add_argument('-opt', type=str, default='SGD', help='Optimizer (SGD or AdamW)')
    parser.add_argument('-lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='Learning rate scheduler')
    parser.add_argument('-step_size', default=100, type=float, help='Step size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='Gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay')
    
    # Loss Function
    parser.add_argument('-loss_lambda', type=float, default=0.05, help='Scaling factor for MSE term in loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='Use one-hot encoding for MSE target')
    parser.add_argument('-loss_means', type=float, default=1.0, help='Target value for active neurons in MSE loss')
    
    # Misc
    parser.add_argument('-resume', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('-amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('-save_init', action='store_true', help='Save initial parameters')

    args = parser.parse_args()
    print(args)

    # Set seeds
    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)  
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)

    # Data Preparation (Tiny ImageNet)
    print("Loading Tiny ImageNet...")
    c_in = 3
    num_classes = 200 
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Augmentation strategy
    transform_list = [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(0.5),
        autoaugment.TrivialAugmentWide(),     
        transforms.ToTensor(),
        normalize,
    ]
    transform_train = transforms.Compose(transform_list)

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    try:
        trainset = TinyImageNet(root=args.data_dir, train=True, transform=transform_train)
        train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j, pin_memory=True)

        testset = TinyImageNet(root=args.data_dir, train=False, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, pin_memory=True)

    except FileNotFoundError as e:
        print("ERROR: Tiny ImageNet data not found or 'val' folder not preprocessed.")
        raise e

    print(f"Data loaded: {len(trainset)} train samples, {len(testset)} test samples.")
    
    # Model Preparation
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate_self.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()
    else:
        raise NotImplementedError

    neuron_model = neuron.BPTTNeuron

    if 'vgg' in args.model:
        net = spiking_vgg_bn.__dict__[args.model](
            neuron=neuron_model, 
            num_classes=num_classes, 
            neuron_dropout=args.drop_rate, 
            tau=args.tau, 
            surrogate_function=surrogate_function, 
            c_in=c_in, 
            T=args.T
        )
        print(f'Using {args.model} model.')
    else:
        raise NotImplementedError(f"Model {args.model} not supported in this script. Only VGG models are.")

    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()

    # Optimizer & Scheduler
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = torch.amp.GradScaler()

    # Resume from Checkpoint
    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        print('resuming...')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print('start epoch:', start_epoch, ', max test acc:', max_test_acc)


    # Output Directories
    out_dir = os.path.join(args.out_dir, f'TATCA_{args.dataset}_{args.model}_{args.name}_T{args.T}_e{args.epochs}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print('out dir already exists:', out_dir)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    # Training Loop
    criterion_mse = nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        # Train 
        start_time = time.time()
        net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(train_data_loader))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0

        for i, (frame, label) in enumerate(train_data_loader):
            data_time.update(time.time() - end)
            batch_idx += 1
            
            frame = frame.float().cuda()
            label = label.cuda()

            optimizer.zero_grad()


            intermediate_features = []
            for t in range(args.T):

                input_frame_t = frame
                feature_t = net.part1(input_frame_t)
                intermediate_features.append(feature_t)


            sequence = torch.stack(intermediate_features, dim=0)
            attended_sequence = net.tatca_module(sequence)


            total_loss_tensor = 0
            total_fr_for_acc = 0  

            for t in range(args.T):
                attended_frame_t = attended_sequence[t]
                out_fr = net.part2(attended_frame_t)

                if t == 0:
                    total_fr_for_acc = out_fr.clone().detach()
                else:
                    total_fr_for_acc += out_fr.clone().detach()

                with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                    if args.loss_lambda > 0.0:
                        if args.mse_n_reg:
                            label_one_hot = F.one_hot(label, num_classes).float()
                        else:
                            label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss_t = (1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss
                    else:
                        loss_t = F.cross_entropy(out_fr, label)
                    total_loss_tensor = total_loss_tensor + loss_t


            final_loss = total_loss_tensor / args.T

            if args.amp:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                final_loss.backward()
                optimizer.step()


            prec1, prec5 = accuracy(total_fr_for_acc.data, label.data, topk=(1, 5))
            losses.update(final_loss.item(), label.size(0))
            top1.update(prec1.item(), label.size(0))
            top5.update(prec5.item(), label.size(0))

            train_samples += label.numel()
            train_loss += final_loss.item() * label.numel()
            train_acc += (total_fr_for_acc.argmax(1) == label).float().sum().item()

            functional.reset_net(net.part1)
            functional.reset_net(net.part2)
            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                             batch=batch_idx, size=len(train_data_loader), data=data_time.avg, bt=batch_time.avg,
                             total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg, top5=top5.avg)
            bar.next()
        bar.finish()

        train_loss /= train_samples
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        # Test 
        net.eval()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        
        bar = Bar('Testing', max=len(test_data_loader))
        
        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0

        with torch.no_grad():
            for i, (frame, label) in enumerate(test_data_loader):
                batch_idx += 1
                
                frame = frame.float().cuda()
                label = label.cuda()

                intermediate_features = []
                for t in range(args.T):
                    input_frame_t = frame
                    feature_t = net.part1(input_frame_t)
                    intermediate_features.append(feature_t)

                sequence = torch.stack(intermediate_features, dim=0)
                attended_sequence = net.tatca_module(sequence)

                total_fr = 0
                total_loss_item = 0 
                for t in range(args.T):
                    attended_frame_t = attended_sequence[t]
                    out_fr = net.part2(attended_frame_t)

                    if t == 0:
                        total_fr = out_fr
                    else:
                        total_fr += out_fr

                    if args.loss_lambda > 0.0:
                        if args.mse_n_reg:
                            label_one_hot = F.one_hot(label, num_classes).float()
                        else:
                            label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss)
                    else:
                        loss = F.cross_entropy(out_fr, label)
                    total_loss_item += loss.item()

                final_loss_item = total_loss_item / args.T
                
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(final_loss_item, label.size(0)) 
                top1.update(prec1.item(), label.size(0))      
                top5.update(prec5.item(), label.size(0))      

                test_samples += label.numel()
                test_loss += final_loss_item * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()
                
                functional.reset_net(net.part1)
                functional.reset_net(net.part2)
                
                batch_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                             batch=batch_idx, size=len(test_data_loader), data=data_time.avg, bt=batch_time.avg,
                             total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg, top5=top5.avg)
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        # Save Checkpoint 
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
        checkpoint = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'max_test_acc': max_test_acc}
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}, total_time={total_time:.2f}s, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch -1))).strftime("%Y-%m-%d %H:%M:%S")}, current_time={datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print("after one epoch: %.4fGB" % (torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024))


if __name__ == '__main__':
    main()