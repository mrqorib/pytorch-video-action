from datetime import datetime
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import *
from statistics import mode

_TARGET_PAD = -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', dest='pretrained_model', nargs='+', required=True,
                        help='pretrained_model filename, filename must be standard ${model}_${accuracy}_dev, priority is given based on the asc order')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=True,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')
    parser.add_argument("--prob", dest='prob', required=True, choices=['small', 'big'],
                        help='probability smaller or bigger better')
    parser.add_argument("--part", dest='part', default='test', choices=['dev', 'test'],
                    help='infer the dev or test')
    parser.add_argument('--split', dest='split', type=int, default=0,
                        help='split')
    return parser.parse_args()

def pad_batch(batch, batchsize=1):
    batch = list(zip(*batch))
    x, y = batch[0], batch[1]
    x_len = [p.shape[0] for p in x]
    max_length = max(x_len)
    padded_seqs = torch.zeros((batchsize, max_length, 400))
    padded_target = torch.empty((batchsize, max_length), dtype=torch.long).fill_(_TARGET_PAD)
    for i, l in enumerate(x_len):
        padded_seqs[i, 0:l] = x[i][0:l]
        if len(y[i]) > 0:
            padded_target[i, 0:l] = y[i][0:l]
    target = torch.flatten(padded_target)
    return padded_seqs, x_len, target

def most_frequent(List):
    return max(set(List), key = List.count)

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq

def main():
    args = parse_arguments()
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used: {}'.format(device))
    if args.part == 'dev':
        split = args.split
        mode = 'active'
    else:
        split = 1
        mode = None
    test_dataset = VideoDataset(part=args.part, load_all=True, split=split, mode=mode)
    class_info = test_dataset.get_class_info()
    n_class = len(class_info['class_names'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=(lambda x: pad_batch(x, 1)))
    models = {}
    for model_filename in args.pretrained_model:
        model = '_'.join(model_filename.split('.')[0].split('_')[:-1])
        if model == 'vanillalstm':
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
            model_path = os.path.join('models', '{}.pth'.format(model_filename))
            model_state_dict = torch.load(model_path, map_location=device)
            net.load_state_dict(model_state_dict)
            net.to(device)
            # net.eval()
            models[model_filename] = net
            print('Load pretrained model: {}'.format(model_filename))
        except Exception as e:
            print(e)
            print('Model {} not found in {} folder!'.format(model_filename, model_path))
    if(len(models) == 0):
        print('No model is loaded...')
        return 0
    print('Start predicting...')
    results = []
    correct_segment = 0
    total_segment = 0
    for i, data in enumerate(test_loader, 0):
        if i % 10 == 0: print('{} out of {}'.format(i, len(test_dataset)))
        inputs, inputs_len, labels = data
        inputs = inputs.to(device)
        label_seq, length_seq = get_label_length_seq(labels)
        # store as models_results[segment] = [model1, model2, model3...]
        models_result = {}
        for key in models:
            model = models[key]
            outputs = model(inputs, inputs_len)
            _, predicted = torch.max(outputs.data, 1)
            if args.part == 'dev':
                segments = length_seq
            else:
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

                # take next highest if prediction is 0
                if (model_prediction == 0 and torch.bincount(predicted_labels).shape[0] > 1):
                    model_prediction = int(torch.argsort(torch.bincount(predicted_labels))[1].item())

                # if all 0s, ignore the result
                if (model_prediction != 0):
                    model_probability_index = torch.LongTensor([i for i, e in enumerate(predicted_labels) if int(e) == model_prediction]).to(device)
                    model_probability = float(torch.take(normalized_outputs, model_probability_index).mean())
                    models_result[segment_key]['label'].append(model_prediction)
                    models_result[segment_key]['probability'].append(model_probability)
                    models_result[segment_key]['no_of_frames'].append(len(model_probability_index))

        # select the most frequent one out of the model, where first model has the priority
        for seg_index, segment in enumerate(models_result):
            # taking the mode, but if its equal, take either highest/smallest probability
            try:
                label = mode(models_result[segment]['label'])
            except:
                try:
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
                except:
                    # error: every model is predicting 0
                    print('Blank prediction.')
                    label = 0

            if args.part == 'dev':
                try:
                    if label_seq[seg_index].item() == int(label): 
                        correct_segment += 1
                except Exception as e:
                    print(e)
                    print('label_seq: ', label_seq)
                    print('model_result: ', models_result)
                    print('index: ', index)
            else:
                results.append(label)
        total_segment += len(label_seq)
    if args.part == 'dev':
        print('Accuracy: ', 100 * correct_segment / total_segment)
    else:
        result_path = './results/result_{}_{}'.format('_'.join(args.pretrained_model) ,datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        print('Writing results to {}...'.format(result_path))
        lines = 'Id,Category\n'
        for index, result in enumerate(results):
            if(index == len(results) - 1):
                lines += str(index) + ',' + str(result)
            else:
                lines += str(index) + ',' + str(result) + '\n'
        with open(result_path, 'w') as wr:
            wr.writelines(lines)

if __name__ == "__main__":
    main()
