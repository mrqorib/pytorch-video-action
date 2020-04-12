from datetime import datetime
from collections import OrderedDict
import argparse
import os
import sys
import kenlm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import SimpleFC, vanillaLSTM, BiLSTM, BiGRU, MultiHeadAttention, MultiStageModel #TODO: import your model here
from statistics import mode 

_TARGET_PAD = -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', dest='pretrained_model', nargs='+', required=True,
                        help='pretrained_model filename, filename must be standard ${model}_${accuracy}_dev, priority is given based on the asc order')
    parser.add_argument("--part", dest='part', default='test', choices=['dev', 'test'],
                        help='infer the dev or test')
    parser.add_argument('--lm_path', dest='lm_path', default=None,
                        help='Path to the language model for beam search decoding')
    parser.add_argument('--beam_size', dest='beam_size', type=int, default=5,
                        help='beam_size')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.2,
                        help='frame appearance threshold')
    parser.add_argument('--split', dest='split', type=int, default=0,
                        help='split')
    parser.add_argument("--remove_zero", type=bool, nargs='?',
                    const=True, default=False,
                    help='Force zero removal from prediction')
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
    lm_model = kenlm.LanguageModel(args.lm_path)
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
            model_path = os.path.join('.', 'models', '{}.pth'.format(model_filename))
            model_state_dict = torch.load(model_path, map_location=device)
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
    correct_segment = 0
    total_segment = 0
    for i, data in enumerate(test_loader, 0):
        if i % 10 == 0: print('{} out of {}'.format(i, len(test_dataset)))
        inputs, inputs_len, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        label_seq, length_seq = get_label_length_seq(labels)
        pred_probs = []
        pred_classes = []
        for key, model in models.items():
            outputs = model(inputs, inputs_len)
            pred_prob, pred_class = torch.max(outputs.data, 1)
            pred_probs.append(pred_prob)
            pred_classes.append(pred_class)

        if args.part == 'dev':
            segments = length_seq
        else:
            segments = test_dataset.segment_lines[i]
        
        prediction_beam = [('',0)]
        for index, segment in enumerate(segments):
            if (index == len(segments) - 1):
                break
            start_frame = int(segments[index])
            end_frame = int(segments[index+1])
            label_candidates = []
            for predicted in pred_classes: #add prediction from all models
                frame_prediction = predicted[start_frame: end_frame]
                label_count = torch.bincount(frame_prediction)
                label_prob = (label_count - min(label_count)) / (10e-6 + max(label_count) - min(label_count))
                predicted_labels = torch.argsort(label_count, descending=True)
                label_prob = label_prob[predicted_labels]
                mask = label_prob > args.threshold
                predicted_labels = predicted_labels[mask]
                label_candidates.append(predicted_labels)
            label_candidates = torch.cat(label_candidates)
            label_candidates = torch.unique(label_candidates)
            if args.remove_zero:
                label_candidates = label_candidates[label_candidates.nonzero()]
                if len(label_candidates) == 0:
                    label_candidates = torch.tensor([0])

            new_beam = []
            for (current_pred, current_prob) in prediction_beam:
                for label in label_candidates:
                    label = label.item()
                    new_pred = current_pred + ' ' + str(label)
                    new_pred = new_pred.strip()
                    new_prob = lm_model.score(new_pred)
                    new_beam.append((new_pred, new_prob))
            prediction_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:args.beam_size]

        prediction = prediction_beam[0][0].split(' ')
        if args.part == 'dev':
            assert len(prediction) == len(label_seq)
            for index, predicted_label in enumerate(prediction):
                if label_seq[index].item() == int(predicted_label): 
                        correct_segment += 1
            total_segment += len(label_seq)
        else:
            results += prediction

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
        print("Finished! Let's hope it gets better result!")

if __name__ == "__main__":
    main()