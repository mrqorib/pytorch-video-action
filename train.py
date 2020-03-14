from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset
from networks import SimpleFC #TODO: import your model here

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
                        choices=['simple_fc'], #TODO: add your model name here
                        help='Choose the type of model for learning')
    parser.add_argument("--load_all", type=bool, nargs='?',
                        const=True, default=False,
                        help='Load all data into RAM '\
                            '(make sure you have enough free Memory).')                    
    return parser.parse_args()

def evaluate(model, dev_dataset, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dev_dataset:
            inputs = torch.squeeze(data[0], dim=0).to(device)
            labels = torch.squeeze(data[1], dim=0).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100 * correct / total)

def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = VideoDataset(part='train', load_all=args.load_all)
    dev_dataset = VideoDataset(part='dev', load_all=args.load_all)
    class_info = train_dataset.get_class_info()
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize,
                        shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=1,
                        shuffle=False, num_workers=args.num_workers)
    n_class = len(class_info['class_names'])

    if args.model == 'simple_fc':
        net = SimpleFC(400, n_class).to(device)
    #TODO: add your model name here
    # elif args.model == 'my_model':
    #    net = MyNet(<arguments>).to(device)
    else:
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    total_epoch = args.epoch
    
    previous_dev = 0
    for epoch in range(total_epoch):
        start = datetime.now()
        running_loss = 0.0
        print('Starting Epoch #{}, {} iterations'.format(epoch + 1, len(train_loader)))
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0], dim=0).to(device)
            labels = torch.squeeze(data[1], dim=0).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
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