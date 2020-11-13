import argparse
import sys
import os
import shutil
import torch
from torch.optim.lr_scheduler import MultiStepLR
from data import Dataset
from model import MolGen
from rdkit import Chem

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
#parser.add_argument('--batch_size', type=int, default=128)
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

    dataloader = Dataset('./CHEMBL/chembl_27_smiles.txt')

    # build model
    ntoken = 51
    model = MolGen(ntoken, args.hid_size, ntoken,
                   args.embed_size, args.nlayer, args.dropout)
    model = torch.load('./model.pt')
    model.eval()
    
    with torch.no_grad():
        max_len = 50
        temperature = 0.5
        generated = 'X'
        inp = torch.tensor([[dataloader.dictionary.char2idx[generated]]], dtype=torch.long)
        hidden = model.init_hidden(1)
        # start generating new SMILES
        while True:
            output, hidden = model(inp, hidden)
            output_dist = output.view(-1).data.div(temperature).exp()
            next_idx = torch.multinomial(output_dist, 1)
            next_char = dataloader.dictionary.idx2char[next_idx.data]
            generated += next_char
            if next_char == 'Y' or len(generated) >= max_len:
                break
            inp = next_idx.view(1, -1)

        print(generated)
    
    generated = generated[1:].replace('Y', '')
    print(generated)
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(generated))
    print(smiles)
    print(Chem.MolFromSmiles(smiles))

if __name__ == "__main__":
    main()


