from datetime import datetime
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset, BucketBatchSampler
from networks import SimpleFC, vanillaLSTM, BiLSTM, BiGRU, MultiHeadAttention, MultiStageModel #TODO: import your model here


_TARGET_PAD = -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=1, help='learning minibatch size')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10,
                        help='epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='Num of workers to load the dataset. Use 0 for Windows')
    parser.add_argument('--model', dest='model', default='simple_fc',
                        choices=['simple_fc', 'vanilla_lstm', 'bilstm', 'bigru', 'attn', 'ms_tcn'], #TODO: add your model name here
                        help='Choose the type of model for learning')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default=None,
                        help='pretrained_model file name')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=False,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')
    # attn model params
    parser.add_argument('--attn_head', dest='attn_head', type=int, default=4,
                        help='Number of head in MultiHeadAttention')
    return parser.parse_args()

def evaluate(model, dev_dataset, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dev_dataset:
            inputs, inputs_len, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, inputs_len)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100 * correct / total)

def main():
    args = parse_arguments()
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pad_batch(batch, batchsize=args.batchsize):
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

    train_dataset = VideoDataset(part='train', load_all=args.load_all)
    dev_dataset = VideoDataset(part='dev', load_all=args.load_all)
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
        net = vanillaLSTM(400, n_class=n_class).to(device)
    elif args.model == 'bilstm':
       net = BiLSTM(400, n_class=n_class).to(device)
    elif args.model == 'bigru':
       net = BiGRU(400, n_class=n_class).to(device)
    elif args.model == 'attn':
        net = MultiHeadAttention(400, args.attn_head, n_class=n_class).to(device)
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
    criterion = nn.CrossEntropyLoss(ignore_index=_TARGET_PAD)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
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

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, inputs_len)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().item()

        delta_time = (datetime.now() - start).seconds / 60.0
        print('[%d, %5d] loss: %.3f (%.3f mins)' %
            (epoch + 1, i + 1, running_loss / i, delta_time))
        running_loss = 0.0
        dev_acc = evaluate(net, dev_loader, device)
        print('Dev accuracy: {:.3f}'.format(dev_acc))
        if dev_acc > previous_dev:
            print('{} ==> {}'.format(dev_acc, previous_dev))
            model_path = 'models/{}_{:.2f}_dev.pth'.format(args.model, dev_acc)
            torch.save(net.state_dict(), model_path)
            previous_dev = dev_acc

    print('Finished Training, Dev Accuracy: ', previous_dev)


if __name__ == "__main__":
    main()
