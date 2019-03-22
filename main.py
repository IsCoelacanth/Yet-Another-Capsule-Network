import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader

from Capsnet import CapsuleNetwork, CapsuleLoss
import numpy as np
import matplotlib.pyplot as plt

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def imshow(img, labels):
    # print(img.size())
    img = torchvision.utils.make_grid(img, nrow=4)
    img = img.cpu().detach().numpy()
    print('\n', ' '.join('%5s'% classes[j] for j in labels), '\n')
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()


if __name__ == '__main__':
    model = CapsuleNetwork()

    transforms = torchvision.transforms.Compose([\
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.STL10('data', 'train', transform=transforms, download=True)
    test_set = torchvision.datasets.STL10('data', 'test', transform=transforms, download=True)
    # train_set = torchvision.datasets.FakeData(1000, image_size=(3,64,64), transform=transforms)
    # test_set = torchvision.datasets.FakeData(1000, image_size=(3,64,64), transform=transforms)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True, drop_last=True)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = CapsuleLoss()

    model.to(dev)
    loss.to(dev)

    terror = []
    for e in range(10):
        lavg = 0
        loader = tqdm.tqdm(train_loader, total=len(train_loader))

        for i, data in enumerate(loader):
            if i >= 128:
                loader.close()
                break
            img, label = data
            img = img.to(dev)
            label = label.to(dev)

            optim.zero_grad()
            y = model(img, label)
            l = loss(img, label, y[0], y[1])
            lavg += l.item()
            if i > 0:
                lavg /= 2
            l.backward()
            terror.append(l.item())
            optim.step()
            loader.set_description(
                f'loss: {lavg}'
            )
            loader.refresh()
        loader.close()
        plt.plot(list(range(len(terror))), terror)
        plt.show()
        print("testing")
        voader = tqdm.tqdm(test_loader, total=len(test_loader))

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(voader):
                if i >= 64:
                    voader.close()
                    break
                img, labels = data
                img = img.to(dev)
                labels = labels.to(dev)
                y, recon = model(img, labels)
                if i == 0:
                    imshow(img, labels)
                    imshow(recon, labels)
                _, pred = torch.max(y.data, 1)
                print(pred)
                total += len(labels.size())
                correct += (pred == labels).sum().item()
        voader.close()
        print(f'accuracy: {100 * (correct/total)} \%')




