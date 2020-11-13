import argparse
import sys
import os
import shutil
import torch
from torch.optim.lr_scheduler import MultiStepLR
from data import Dataset
from model import MolGen

parser = argparse.ArgumentParser(description='Molecular Generator Model')
parser.add_argument('--root', metavar='DATA_DIR', default='./CHEMBL/chembl_27_smiles.txt')
# hyper parameter tuning
parser.add_argument('--nlayer', type=int, default=3)
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hid_size', type=int, default=256)
parser.add_argument('--seq_len', type=int, default=52)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--lr_milestones', nargs='+', type=int, default=[10, 20, 40])
parser.add_argument('--stn', action='store_true')
# default params
parser.add_argument('--start_epoch', type=int, metavar='N', default=0)
parser.add_argument('--weight_decay', type=float, metavar='W', default=0)
parser.add_argument('--momentum', type=float, metavar='M', default=0.9)
n_threads = torch.get_num_threads()
parser.add_argument('--num_threads', type=int, metavar='N_thread', default=n_threads)
parser.add_argument('--print_freq', type=int, metavar='N', default=100)
parser.add_argument('--save', type=str, default='model.pt')

# parse args
args = parser.parse_args()
if args.num_threads != n_threads:
    torch.set_num_threads(args.num_threads)
print('User defined variables:', flush=True)
for key, val in vars(args).items():
    print('  => {:17s}: {}'.format(key, val), flush=True)

def main():
    global args

    # load dataset
    print('loading dataset...', flush=True)
    dataloader = Dataset('./CHEMBL/chembl_27_smiles.txt')
    data = dataloader.dataset.view(-1, args.seq_len)
    print('finished loading data!\n', flush=True)
    
    # batchify
    nbatch = data.size(0) // args.batch_size
    data = data.narrow(0, 0, args.batch_size*nbatch)

    # build model
    ntoken = len(dataloader.dictionary)
    print('number of tokens:', ntoken, flush=True)
    print(dataloader.dictionary.char2idx.keys(), flush=True)
    
    model = MolGen(ntoken, args.hid_size, ntoken,
                   args.embed_size, args.nlayer, args.dropout)
    
    # info about trainable model parameters
    trainable_params = sum(p.numel() for p in model.parameters() 
                           if p.requires_grad)
    print('Number of trainable model parameters: {:d}' \
           .format(trainable_params), flush=True)
    for ip in model.parameters():
        print('shape: {:<30}size: {}'.format(str(ip.shape), str(ip.numel())), flush=True)
    
    # define loss function 
    criterion = torch.nn.CrossEntropyLoss()

    # optimization algo
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD is currently allowed as --optim')

    # learning-rate scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.lr_milestones,
                            gamma=0.1, last_epoch=-1)
    
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        print('Epoch {}'.format(epoch), flush=True)
        # train for one epoch
        train(data, model, criterion, optimizer)

        # save current model
        with open(args.save, 'wb') as f:
            torch.save(model, f)

        scheduler.step()


def train(data, model, criterion, optimizer):
    # switch to training mode
    model.train()

    hidden = model.init_hidden(args.batch_size)

    running_loss = 0
    for ibatch in range(data.size(0)//args.batch_size):
        idata = data[ibatch*args.batch_size:(ibatch+1)*args.batch_size]
        
        source = idata[:,:-1].transpose(0, 1)
        target = idata[:,1:].transpose(0, 1)

        loss = 0
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach()
        else:
            hidden = tuple(h.detach() for h in hidden)

        output, hidden = model(source, hidden)
        for ichar in range(source.size(0)):
            loss += criterion(output[ichar,:], target[ichar,:])

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress and write to TensorBoard
        running_loss += loss.item()
        if (ibatch+1) % args.print_freq == 0:
            print('running loss:', running_loss/args.print_freq, flush=True)
            running_loss = 0.0


if __name__ == "__main__":
    main()


