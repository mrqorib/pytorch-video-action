from datetime import datetime
import argparse
import os
import sys
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import SimpleFC, vanillaLSTM, BiLSTM, BiGRU, MultiHeadAttention, MultiStageModel #TODO: import your model here
from statistics import mode 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', dest='pretrained_model', nargs='+', required=True,
                        help='pretrained_model filename, filename must be standard ${model}_${accuracy}_dev, priority is given based on the asc order')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=False,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')
    parser.add_argument("--prob", dest='prob', required=True, choices=['small', 'big'],
                        help='probability smaller or bigger better')
    return parser.parse_args()

def pad_batch(batch, batchsize=1):
    batch = list(zip(*batch))
    x, y = batch[0], batch[1]
    x_len = [p.shape[0] for p in x]
    max_length = max(x_len)
    padded_seqs = torch.zeros((batchsize, max_length, 400))
    for i, l in enumerate(x_len):
        padded_seqs[i, 0:l] = x[i][0:l]
    return padded_seqs, x_len

def most_frequent(List): 
    return max(set(List), key = List.count) 

def main():
    args = parse_arguments()
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used: {}'.format(device))
    test_dataset = VideoDataset(part='test', load_all=args.load_all, split=1)
    class_info = test_dataset.get_class_info()
    n_class = len(class_info['class_names'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=(lambda x: pad_batch(x, 1)))
    models = {}
    for model_filename in args.pretrained_model:
        model = '_'.join(model_filename.split('.')[0].split('_')[:-1])
        if model == 'simple_fc':
            net = SimpleFC(400, n_class).to(device)
        elif model == 'vanilla_lstm':
            net = vanillaLSTM(400, n_class=n_class).to(device)
        elif model == 'bilstm':
            net = BiLSTM(400, n_class=n_class).to(device)
        elif model == 'bigru':
            net = BiGRU(400, n_class=n_class).to(device)
        elif model == 'attn':
            net = MultiHeadAttention(400, args.attn_head, n_class=n_class).to(device)
        elif model == 'mstcn':
            net = MultiStageModel(400, n_class=n_class).to(device)
        try:
            model_state_dict = torch.load(os.path.join('.', 'models', '{}.pth'.format(model_filename)))
            net.load_state_dict(model_state_dict)
            net.to(device)
            net.eval()
            models[model_filename] = net
            print('Load pretrained model: {}'.format(model_filename))
        except:
            print('Model {} not found in ./models folder!'.format(model_filename))
    if(len(models) == 0):
        print('No model is loaded...')
        return 0
    print('Start predicting...')
    results = []
    for i, data in enumerate(test_loader, 0):
        if i % 10 == 0: print('{} out of {}'.format(i, len(test_dataset)))
        inputs, inputs_len = data
        inputs = inputs.to(device)
        # store as models_results[segment] = [model1, model2, model3...]
        models_result = {}
        for key in models:
            model = models[key]
            outputs = model(inputs, inputs_len)
            _, predicted = torch.max(outputs.data, 1)
            segments = test_dataset.segment_lines[i]
            # break the predicted labels to segments and take max frequency
            for index, segment in enumerate(segments):
                if (index == len(segments) - 1):
                    break
                start_frame = int(segments[index])
                end_frame = int(segments[index+1])
                segment_key = str(start_frame) + '-' + str(end_frame)
                if(segment_key not in models_result): 
                    models_result[segment_key] = {
                        'label': [],
                        'probability': [],
                        'no_of_frames': []
                    }
                predicted_labels = predicted[start_frame: end_frame]
                normalized_outputs = (_[start_frame: end_frame])/sum(_)
                # get most frequent one
                model_prediction = int(torch.argmax(torch.bincount(predicted_labels)).item())
                model_probability_index = torch.LongTensor([i for i, e in enumerate(predicted_labels) if int(e) == model_prediction]).to(device)
                model_probability = float(torch.take(normalized_outputs, model_probability_index).mean())
                models_result[segment_key]['label'].append(model_prediction)
                models_result[segment_key]['probability'].append(model_probability)
                models_result[segment_key]['no_of_frames'].append(len(model_probability_index))
        # select the most frequent one out of the model, where first model has the priority
        for segment in models_result:
            # taking the mode, but if its equal, take either highest/smallest probability
            try:
                label = mode(models_result[segment]['label'])
            except:
                # if same length then get the probability
                if(len(set(models_result[segment]['no_of_frames'])) == 1):
                    probability = models_result[segment]['probability']
                    if args.prob == 'big':
                        index = probability.index(max(probability))
                    else:
                        index = probability.index(min(probability))
                else:
                    no_of_frames = models_result[segment]['no_of_frames']
                    index = no_of_frames.index(max(no_of_frames))
                label = models_result[segment]['label'][index]
            results.append(label)             
    result_path = './results/result_{}_{}'.format('_'.join(args.pretrained_model) ,datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
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
