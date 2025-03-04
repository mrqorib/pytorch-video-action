from datetime import datetime
import argparse
import os

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
                                 'bigru', 'attn', 'ms_tcn', 'ctcloss'], #TODO: add your model name here
                        help='Choose the type of model for learning')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default=None,
                        help='pretrained_model file name')
    parser.add_argument('--train_mode', dest='train_mode', default='active',
                        choices=['segment', 'active', 'cont'],
                        help='Choose the training mode:\n'\
                             '  > segment: one training instance contains only 1 segment'\
                             '  > active: one training instance is a video with the SIL frames removed'\
                             '  > cont: train the video as whole contiguously')
    parser.add_argument('--pred_mode', dest='pred_mode', default='cont',
                        choices=['last', 'avg', 'cont'], help='Classification for segment train-mode')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=True,
                        help='[Deprecated ]Now enforced to use --load_all')
    parser.add_argument("--eval", type=bool, nargs='?',
                        const=True, default=False,
                        help='Only evaluating model, not training')
    # beam search params
    parser.add_argument('--lm_path', dest='lm_path', default=None,
                        help='Path to the language model for beam search decoding')
    parser.add_argument('--beam_size', dest='beam_size', type=int, default=5,
                        help='beam_size')
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
    parser.add_argument('--lstm_hidden2', dest='lstm_hidden2', type=int, default=64,
                        help='Number of linear hidden neuron')
    return parser.parse_args()

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

def eval_beam_search(model, dev_dataset, device, lm_path,
                beam_size=5, threshold=0.15):
    import kenlm
    lm_model = kenlm.LanguageModel(lm_path)

    model.eval()
    correct_segment = 0
    total_segment = 0
    correct_frame = 0
    total_frame = 0
    for data in dev_dataset:
        inputs, inputs_len, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        label_seq, length_seq = get_label_length_seq(labels)

        outputs = model(inputs, inputs_len)
        _, predicted = torch.max(outputs.data, 1)
        total_frame += labels.size(0)
        correct_frame += (predicted == labels).sum().item()
        
        prediction_beam = [('',0)]
        for index, segment in enumerate(length_seq):
            if (index == len(length_seq) - 1):
                break
            start_frame = int(length_seq[index])
            end_frame = int(length_seq[index+1])
            frame_prediction = predicted[start_frame: end_frame]
            # add to the beam
            label_count = torch.bincount(frame_prediction)
            label_prob = (label_count - min(label_count)) / (10e-6 + max(label_count) - min(label_count))
            predicted_labels = torch.argsort(label_count, descending=True)
            label_prob = label_prob[predicted_labels]
            mask = label_prob > threshold
            predicted_labels = predicted_labels[mask]
            new_beam = []
            for (current_pred, current_prob) in prediction_beam:
                for label in predicted_labels:
                    label = label.item()
                    new_prob = current_prob
                    new_pred = current_pred + ' ' + str(label)
                    new_pred = new_pred.strip()
                    lm_score = lm_model.score(new_pred)
                    new_prob = lm_score
                    new_beam.append((new_pred, new_prob))
            prediction_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
        prediction = prediction_beam[0][0].split(' ')
        assert len(prediction) == len(label_seq)
        for index, predicted_label in enumerate(prediction):
            if label_seq[index].item() == int(predicted_label): 
                    correct_segment += 1
        total_segment += len(label_seq)
        # break
    
    accuracy_frame = (100 * correct_frame / total_frame)
    accuracy_segment = (100 * correct_segment / total_segment)
    return accuracy_segment, accuracy_frame 

def evaluate(model, dev_dataset, device):
    model.eval()
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
            _, predicted = torch.max(outputs.data, 1)
            total_frame += labels.size(0)
            correct_frame += (predicted == labels).sum().item()

            for index, segment in enumerate(length_seq):
                if (index == len(length_seq) - 1):
                    break
                start_frame = int(length_seq[index])
                end_frame = int(length_seq[index+1])
                predicted_labels = predicted[start_frame: end_frame]
                # get most frequent one
                predicted_label = int(torch.argmax(torch.bincount(predicted_labels)).item())
                if label_seq[index] == predicted_label:
                    correct_segment += 1

            total_segment += len(label_seq)

    accuracy_frame = (100 * correct_frame / total_frame)
    accuracy_segment = (100 * correct_segment / total_segment)
    return accuracy_segment, accuracy_frame

def main():
    args = parse_arguments()
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pad_batch(batch, batchsize=args.batchsize, mode=args.pred_mode):
            batch = list(zip(*batch))
            x, y = batch[0], batch[1]
            x_len = [p.shape[0] for p in x]
            max_length = max(x_len)
            padded_seqs = torch.zeros((batchsize, max_length, 400))
            if mode != 'cont':
                y_length = 1
            else:
                y_length = max_length
            padded_target = torch.empty((batchsize, y_length), dtype=torch.long).fill_(_TARGET_PAD)
            for i, l in enumerate(x_len):
                padded_seqs[i, 0:l] = x[i][0:l]
                if mode != 'cont':
                    padded_target[i,:] = y[i]
                else:
                    current_y = y[i]
                    if args.train_mode == 'segment':
                        current_y = torch.repeat_interleave(current_y, l)
                    padded_target[i, 0:l] = current_y[0:l]

            target = torch.flatten(padded_target)
            return padded_seqs, x_len, target

    train_dataset = VideoDataset(part='train', load_all=args.load_all, split=args.split, mode=args.train_mode)
    dev_dataset = VideoDataset(part='dev', load_all=args.load_all, split=args.split, mode=args.train_mode)
    class_info = train_dataset.get_class_info()
    bucket_batch_sampler = BucketBatchSampler(train_dataset.features, args.batchsize)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                        batch_sampler=bucket_batch_sampler, collate_fn=pad_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False,
                        collate_fn=(lambda x: pad_batch(x, 1)),
                        num_workers=args.num_workers)
    n_class = len(class_info['class_names'])

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
    elif args.model == 'ctcloss':
        net = BiGRU(400, n_class=n_class+1).to(device)
    #TODO: add your model name here
    # elif args.model == 'my_model':
    #    net = MyNet(<arguments>).to(device)
    else:
        raise NotImplementedError

    if args.pretrained_model is not None:
        model_path = os.path.join('models', '{}.pth'.format(args.pretrained_model))
        model_state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(model_state_dict)
    # criterion = nn.CrossEntropyLoss()
    if args.model == 'ms_tcn':
        criterion = nn.CrossEntropyLoss(ignore_index=_TARGET_PAD)
    elif args.model == 'ctcloss':
        criterion = nn.CTCLoss(blank=n_class, zero_infinity=True)
    else:
        criterion = nn.NLLLoss(ignore_index=_TARGET_PAD)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, gamma=args.lr_gamma)
    total_epoch = args.epoch

    if args.eval:
        if args.pretrained_model is None:
            print('[ERROR] Please provide the model path with --pretrained_model <model_path>')
            print('Exiting.')
            return
        
        if args.lm_path is not None:
            dev_acc, frame_acc = eval_beam_search(net, dev_loader, device, args.lm_path, args.beam_size)
        else:
            dev_acc, frame_acc = evaluate(net, dev_loader, device)
        
        print('Dev accuracy by frame: {:.3f}'.format(frame_acc))
        print('Dev accuracy by segment: {:.3f}'.format(dev_acc))
        return

    previous_dev = 0
    for epoch in range(total_epoch):
        net.train()
        start = datetime.now()
        running_loss = 0.0
        print('Starting Epoch #{}, {} iterations'.format(epoch + 1, len(train_loader)))
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, inputs_len, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, inputs_len)

            # if using ctc loss: prepare the inputs
            if args.model == 'ctcloss':
                labels = labels.reshape(inputs.shape[0], -1)
                targets = torch.tensor([], dtype=torch.int64, device=device)
                targets_len = []

                for j in range(labels.shape[0]):
                    unique_labels = torch.unique_consecutive(labels[j])
                    targets = torch.cat((targets, unique_labels))
                    targets_len.append(unique_labels.shape[0])

                outputs = outputs.reshape(inputs.shape[0], inputs.shape[1], -1)
                outputs = outputs.permute(1,0,2)
                loss = criterion(outputs, targets, torch.tensor(inputs_len), torch.tensor(targets_len))

            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().item()

        if args.lr_step_size > 0 and args.lr_gamma < 1:
            lr_scheduler.step()

        delta_time = (datetime.now() - start).seconds / 60.0
        print('[%d, %5d] Train loss: %.3f (%.3f mins)' %
            (epoch + 1, i + 1, running_loss / i, delta_time))
        running_loss = 0.0
        dev_acc, frame_acc = evaluate(net, dev_loader, device)
        print('Dev accuracy by frame: {:.3f}'.format(frame_acc))
        print('Dev accuracy by segment: {:.3f} (Current best: {:.3f})'\
            .format(dev_acc, previous_dev))
        if dev_acc > previous_dev:
            print('{} ==> {}'.format(dev_acc, previous_dev))
            model_path = 'models/{}_{:.2f}_dev.pth'.format(args.model, dev_acc)
            torch.save(net.state_dict(), model_path)
            previous_dev = dev_acc

    print('Finished Training, Dev Accuracy: ', previous_dev)


if __name__ == "__main__":
    main()
