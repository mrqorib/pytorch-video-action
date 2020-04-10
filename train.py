from datetime import datetime
import argparse
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import *

_TARGET_PAD = -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=1, help='learning minibatch size')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10,
                        help='epoch')
    parser.add_argument('--split', dest='split', type=int, default=0,
                        help='split')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_step_size', dest='lr_step_size', type=int, default=30,
                        help='learning rate')
    parser.add_argument('--lr_gamma', dest='lr_gamma', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='Num of workers to load the dataset. Use 0 for Windows')
    parser.add_argument('--model', dest='model', default='simple_fc',
                        choices=['simple_fc', 'vanilla_lstm', 'bilstm',
                                 'bilstm_lm', 'attn', 'win_attn',
                                 'bigru', 'attn', 'ms_tcn'], #TODO: add your model name here
                        help='Choose the type of model for learning')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default=None,
                        help='pretrained_model file name')
    parser.add_argument('--train_mode', dest='train_mode', default='active',
                        choices=['segment', 'active', 'cont'],
                        help='Choose the training mode:\n'\
                             '  > segment: one training instance contains only 1 segment'\
                             '  > active: one training instance is a video with the SIL frames removed'\
                             '  > cont: train the video as whole contiguously')
    parser.add_argument('--label_mode', dest='label_mode', default='active',
                        choices=['single', 'multi'],
                        help='Choose the label mode:\n'\
                             '  > single: 0-48 as the default class id provided by TA'\
                             '  > multi: [0-13][0-27] in the form of [action, object] id')
    parser.add_argument('--pred_mode', dest='pred_mode', default='cont',
                        choices=['last', 'avg', 'cont'], help='Classification for segment train-mode')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=False,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')
    # attn model params
    parser.add_argument('--attn_head', dest='attn_head', type=int, default=4,
                        help='Number of head in MultiHeadAttention')
    # lstm model params
    parser.add_argument('--lstm_layer', dest='lstm_layer', type=int, default=2,
                        help='Number of LSTM layer')
    parser.add_argument('--lstm_dropout', dest='lstm_dropout', type=float, default=0.5,
                        help='Dropout rate of LSTM layer')
    parser.add_argument('--lstm_hidden1', dest='lstm_hidden1', type=int, default=256,
                        help='Number of LSTM Hidden neurons')
    parser.add_argument('--lstm_hidden2', dest='lstm_hidden2', type=int, default=2,
                        help='Number of linear hidden neuron')
    return parser.parse_args()

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    # print(content)
    for i in range(len(content)):
        # if torch.equal(content[i] != content[start]:
        # print('{} {}'.format(content[i], content[start]))
        if not torch.equal(content[i], content[start]):
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq

def evaluate(model, dev_dataset, device, label_mode):
    correct_segment = 0
    total_segment = 0
    correct_frame = 0
    total_frame = 0
    with torch.no_grad():
        for data in dev_dataset:
            inputs, inputs_len, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            label_seq, length_seq = get_label_length_seq(labels)
            outputs = model(inputs, inputs_len)
            if label_mode == 'single':
                _, predicted = torch.max(outputs.data, 1)
            elif label_mode == 'multi':
                action_pred = torch.argmax(outputs[0], dim=1, keepdim=True)
                object_pred = torch.argmax(outputs[1], dim=1, keepdim=True)
                predicted = torch.cat([action_pred, object_pred], dim=1)
                # labels = torch.split(labels, 1, dim=1)
                # if correct_frame == 0:
                #     print('frame\npredicted: {}\nlabels: {}\n'.format(predicted, labels))
            total_frame += labels.size(0)
            correct_frame += (predicted == labels).sum().item()
            
            for index, segment in enumerate(length_seq):
                if (index == len(length_seq) - 1):
                    break
                start_frame = int(length_seq[index])
                end_frame = int(length_seq[index+1])
                predicted_labels = predicted[start_frame: end_frame]
                if label_mode == 'single':
                    # get most frequent one
                    predicted_label = int(torch.argmax(torch.bincount(predicted_labels)).item())
                    if label_seq[index] == predicted_label: 
                        correct_segment += 1
                elif label_mode == 'multi':
                    pred_list = predicted_labels.cpu().numpy()
                    pred = [tuple(p) for p in pred_list]
                    label = tuple(label_seq[index].tolist())
                    segment_pred = Counter(pred).most_common(1)[0][0]
                    # if correct_segment == 0:
                    #     print('segment\npredicted: {}\nlabels: {}\n'.format(segment_pred, label))
                    if segment_pred == label:
                        correct_segment += 1
            total_segment += len(label_seq)

    accuracy_frame = (100 * correct_frame / total_frame)
    accuracy_segment = (100 * correct_segment / total_segment)
    return accuracy_segment, accuracy_frame

def main():
    args = parse_arguments()
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pad_batch(batch, batchsize=args.batchsize, mode=args.pred_mode, class_dim=2):
            batch = list(zip(*batch))
            x, y = batch[0], batch[1]
            x_len = [p.shape[0] for p in x]
            max_length = max(x_len)
            padded_seqs = torch.zeros((batchsize, max_length, 400))
            
            if mode != 'cont':
                y_length = 1
            else:
                y_length = max_length
            if args.label_mode == 'single':
                padded_target = torch.empty((batchsize, y_length), dtype=torch.long).fill_(_TARGET_PAD)
            elif args.label_mode == 'multi':
                padded_target = torch.empty((batchsize, y_length, class_dim), dtype=torch.long).fill_(_TARGET_PAD)
            
            for i, l in enumerate(x_len):
                padded_seqs[i, 0:l] = x[i][0:l]
                if mode != 'cont':
                    padded_target[i,:] = y[i]
                else:
                    current_y = y[i]
                    if args.train_mode == 'segment':
                        current_y = torch.repeat_interleave(current_y, l)
                    padded_target[i, 0:l] = current_y[0:l]

            target = padded_target.flatten(0, 1)
            return padded_seqs, x_len, target
    
    train_dataset = VideoDataset(part='train', load_all=args.load_all,
                                 split=args.split, mode=args.train_mode,
                                 label_mode=args.label_mode)
    dev_dataset = VideoDataset(part='dev', load_all=args.load_all,
                               split=args.split, mode=args.train_mode,
                               label_mode=args.label_mode)
    n_class = train_dataset.get_class_num()
    bucket_batch_sampler = BucketBatchSampler(train_dataset.features, args.batchsize)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                        batch_sampler=bucket_batch_sampler,
                        collate_fn=(lambda x: pad_batch(x, class_dim=len(n_class))))
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False,
                        collate_fn=(lambda x: pad_batch(x, 1)),
                        num_workers=args.num_workers)

    if args.model == 'simple_fc':
        net = SimpleFC(400, n_class).to(device)
    elif args.model == 'vanilla_lstm':
        net = vanillaLSTM(400,
                        lstm_layer=args.lstm_layer,
                        hidden_dim=args.lstm_hidden1,
                        dropout_rate=args.lstm_dropout,
                        n_class=n_class,
                        mode=args.pred_mode).to(device)
    elif args.model == 'bilstm':
        net = BiLSTM(input_dim=400,
                    lstm_layer=args.lstm_layer,
                    hidden_dim_1=args.lstm_hidden1,
                    dropout_rate=args.lstm_dropout,
                    hidden_dim_2=args.lstm_hidden2,
                    n_class=n_class,
                    mode=args.pred_mode).to(device)
    elif args.model == 'bilstm_lm':
        net = BiLSTMWithLM(input_dim=400,
                    lstm_layer=args.lstm_layer,
                    hidden_dim_1=args.lstm_hidden1,
                    dropout_rate=args.lstm_dropout,
                    hidden_dim_2=args.lstm_hidden2,
                    n_class=n_class).to(device)
    elif args.model == 'win_attn':
        net = ExpWindowAttention(400, args.attn_head, n_class=n_class).to(device)
    elif args.model == 'bigru':
       net = BiGRU(400, n_class=n_class).to(device)
    elif args.model == 'attn':
        net = MultiHeadAttention(400,
                                 args.attn_head,
                                 n_class=n_class,
                                 mode=args.pred_mode).to(device)
    elif args.model == 'ms_tcn':
        net = MultiStageModel(400, n_class=n_class).to(device)
    #TODO: add your model name here
    # elif args.model == 'my_model':
    #    net = MyNet(<arguments>).to(device)
    else:
        raise NotImplementedError

    if args.pretrained_model is not None:
        model_path = os.path.join('models', '{}.pth'.format(args.pretrained_model))
        model_state_dict = torch.load(model_path)
        net.load_state_dict(model_state_dict)
    # criterion = nn.CrossEntropyLoss()
    if args.model == 'ms_tcn':
        criterion = nn.CrossEntropyLoss(ignore_index=_TARGET_PAD)
    else:
        criterion = nn.NLLLoss(ignore_index=_TARGET_PAD)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, gamma=args.lr_gamma)
    total_epoch = args.epoch

    previous_dev = 0
    for epoch in range(total_epoch):
        start = datetime.now()
        running_loss = 0.0
        print('Starting Epoch #{}, {} iterations'.format(epoch + 1, len(train_loader)))
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, inputs_len, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # print('inputs: ', inputs)
            # print('inputs len: ', inputs_len)
            # print('labels: ', labels)
            # if i > 3:
            #     break
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, inputs_len)
            # print(outputs.shape)
            # print(labels.shape)
            if args.label_mode == 'single':
                loss = criterion(outputs, labels)
            elif args.label_mode == 'multi':
                labels = torch.split(labels, 1, dim=1)
                loss = 0
                weight = (0.5, 0.5) #dim = (num_class)
                # print('outputs: ', outputs)
                # print('labels: ', labels)
                # print('len out - lab - wgh: {} - {} - {}'\
                #         .format(len(outputs), len(labels), len(weight)))
                # print('len class ', len(n_class))
                for li in range(len(n_class)):
                    loss += weight[li] * criterion(outputs[li], labels[li].squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().item()

        if args.lr_step_size > 0 and args.lr_gamma < 1:
            lr_scheduler.step()

        delta_time = (datetime.now() - start).seconds / 60.0
        print('[%d, %5d] loss: %.3f (%.3f mins)' %
            (epoch + 1, i + 1, running_loss / i, delta_time))
        running_loss = 0.0
        dev_acc, frame_acc = evaluate(net, dev_loader, device, args.label_mode)
        print('Dev accuracy by frame: {:.3f}'.format(frame_acc))
        print('Dev accuracy by segment: {:.3f}'.format(dev_acc))
        if dev_acc > previous_dev:
            print('{} ==> {}'.format(dev_acc, previous_dev))
            model_path = 'models/{}_{:.2f}_dev.pth'.format(args.model, dev_acc)
            torch.save(net.state_dict(), model_path)
            previous_dev = dev_acc

    print('Finished Training, Dev Accuracy: ', previous_dev)


if __name__ == "__main__":
    main()
