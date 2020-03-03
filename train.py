import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import VideoDataset
from networks import SimpleFC


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = VideoDataset(part='train')
    test_dataset = VideoDataset(part='train') #TODO: change later
    class_info = train_dataset.get_class_info()
    train_loader = DataLoader(train_dataset, batch_size=1,
                        shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1,
                        shuffle=True, num_workers=4)
    n_class = len(class_info['class_names'])
    net = SimpleFC(400, n_class).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    total_epoch = 10

    for epoch in range(total_epoch):  # loop over the dataset multiple times
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
            # print(inputs.size())
            # print(labels.size())
            # print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / i))
        running_loss = 0.0

    print('Finished Training')

    PATH = 'models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    net = SimpleFC(400, n_class).to(device)
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            inputs = torch.squeeze(data[0], dim=0).to(device)
            labels = torch.squeeze(data[1], dim=0).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train data: %d %%' % (
    100 * correct / total))


if __name__ == "__main__":
    main()