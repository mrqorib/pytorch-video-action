from datetime import datetime
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import SimpleFC, vanillaLSTM, BiLSTM #TODO: import your model here
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', dest='pretrained_model', help='pretrained_model file name')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=False,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')
    parser.add_argument('--model', dest='model', default='simple_fc',
                        choices=['simple_fc', 'vanilla_lstm', 'bilstm'], #TODO: add your model name here
                        help='Choose the type of model for testing')
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used: {}'.format(device))
        
    def pad_batch(batch, batchsize=1):
            batch = list(zip(*batch))
            x, y = batch[0], batch[1]
            x_len = [p.shape[0] for p in x]
            max_length = max(x_len)
            
            padded_seqs = torch.zeros((batchsize, max_length, 400))
            for i, l in enumerate(x_len):
                padded_seqs[i, 0:l] = x[i][0:l]
            
            return padded_seqs, x_len
    test_dataset = VideoDataset(part='test', load_all=args.load_all, split=1)
    class_info = test_dataset.get_class_info()
    n_class = len(class_info['class_names'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=(lambda x: pad_batch(x, 1)))
    
    if args.model == 'simple_fc':
       net = SimpleFC(400, n_class).to(device)
    elif args.model == 'vanilla_lstm':
       net = vanillaLSTM(400, n_class=n_class).to(device)
    elif args.model == 'bilstm':
       net = BiLSTM(400, n_class=n_class).to(device)
    print('Load pretrained model: {}'.format(args.pretrained_model))
    try:
        model_state_dict = torch.load(os.path.join('.', 'models', '{}.pth'.format(args.pretrained_model)))
        net.load_state_dict(model_state_dict)
        net.to(device)
        net.eval()
    except:
        print('Model not found in ./models folder!')
    print('Start predicting...')
    results = []
    for i, data in enumerate(test_loader, 0):
            inputs, inputs_len = data
            inputs = inputs.to(device)
            # print('inputs: ', inputs)
            outputs = net(inputs, inputs_len)
            _, predicted = torch.max(outputs.data, 1)
            # get most frequent one
            results.append(torch.argmax(torch.bincount(predicted)).item())
    result_path = './results/result_{}_{}'.format(args.pretrained_model ,datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    print('Writing results to {}...'.format(result_path))
    results_df = pd.DataFrame(results)
    results_df.index.name = 'Id'
    results_df.columns = ['Category']
    results_df.to_csv(result_path, mode='a', line_terminator="", sep=',')
    # remove last blank line written by pandas to_csv
    with open(result_path) as f:
        lines = f.readlines()
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(result_path, 'w') as wr:
        wr.writelines(lines)

if __name__ == "__main__":
    main()
