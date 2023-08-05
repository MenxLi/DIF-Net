import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Mixed_CBCT_dataset
from models.model import DIF_Net
from utils import convert_cuda, add_argument
from evaluate import eval_one_epoch
import tqdm
import torch.utils.tensorboard as tb

def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser = add_argument(parser)
    args = parser.parse_args()
    print(args)
    
    save_dir = f'./logs/{args.name}'
    os.makedirs(save_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=save_dir+'/tb')
    
    # -- initialize training dataset/loader
    dst_list = args.dst_list.split('+')
    train_dst = Mixed_CBCT_dataset(
        dst_list=dst_list,
        split='train', 
        num_views=args.num_views, 
        npoint=args.num_points,
        out_res=args.out_res,
        random_views=args.random_views,
    )
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        Mixed_CBCT_dataset(
            dst_list=dst_list,
            split='eval',
            num_views=args.num_views,
            out_res=128, # low-res evaluation is faster
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- initialize model
    model = DIF_Net(
        num_views=args.num_views,
        combine=args.combine,
        mid_ch=128
    ).cuda()
    
    # -- initialize optimizer, lr scheduler, and loss function
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=1e-3
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(0.001, 1 / args.epoch)
    )
    loss_func = nn.MSELoss()

    # -- training starts
    for epoch in range(args.epoch + 1):
        loss_list = []
        model.train()
        for item in tqdm.tqdm(train_loader):  
            optimizer.zero_grad()
            item = convert_cuda(item)
            pred, gt = model(item)
            loss = loss_func(pred, gt)
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if epoch % 1 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(loss_list)))
            writer.add_scalar('loss', np.mean(loss_list), epoch) # add loss
        
        # -- save ckpt
        if epoch % 10 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'ep_{epoch}.pth')
            )

        # -- evaluation
        if epoch % 10 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
            metrics, _ = eval_one_epoch(
                model, 
                eval_loader, 
                args.eval_npoint
            )
            msg = f' --- epoch {epoch}'
            for dst_name in metrics.keys():
                msg += f', {dst_name}'
                met = metrics[dst_name]
                for key, val in met.items():
                    msg += ', {}: {:.4}'.format(key, val)
                    writer.add_scalar(f'{dst_name}/{key}', val, epoch)
            print(msg)
        
        #tensorboard log
        writer.close() # 
        
        lr_scheduler.step()
