import torch
import torchvision
import tqdm
import argparse
from Capsnet import CapsuleNetwork, CapsuleLoss
import numpy as np
import matplotlib.pyplot as plt
from datasets import get_dataset

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def imshow(img, labels, path, cls):
    # img = torchvision.utils.make_grid(img, nrow=4)
    batch = img.size(0)
    row = img.size(0)//4
    if row == 0:
        row = 1
    img = img.cpu().detach().numpy()
    # print('\n', ' '.join('%s'% classes[j] for j in labels))

    for j in range(batch):
        plt.subplot(row, 4, j+1)
        plt.imshow(np.transpose(img[j, :, :, :], (1,2,0)))
        plt.title(classes[labels[j]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path + f'/images/{cls}.jpg', dpi=300)
    plt.show()
    plt.close()


def plot_show(ep_err, av_err, acc, path):
    ep_dim = list(range(len(ep_err)))
    av_dim = list(range(len(av_err)))

    plt.subplot(311)
    plt.plot(ep_dim, ep_err)
    plt.xlabel('batches')
    plt.ylabel('training error')
    plt.title('training error over an epoch')
    plt.subplot(312)
    plt.plot(av_dim, av_err)
    plt.xlabel('epochs')
    plt.ylabel('average training error')
    plt.title('average training per epoch')
    plt.subplot(313)
    plt.plot(list(range(len(acc))), acc)
    plt.xlabel('epochs')
    plt.ylabel('prediction accuracy')
    plt.title('accuracy over epochs')
    plt.tight_layout()
    plt.savefig(path + f'/graphs.jpg', dpi=300)
    plt.show()
    plt.close()


def train(model, train_loader, test_loader, path):

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = CapsuleLoss()

    model.to(dev)
    loss.to(dev)

    terror = []
    tmean = []
    acc = []
    for e in range(50):
        lavg = 0
        loader = tqdm.tqdm(train_loader, total=len(train_loader))

        for i, data in enumerate(loader):
            if i >= 32:
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
        tmean.append(sum(terror)/len(terror))
        # plt.show()
        print("testing")
        voader = tqdm.tqdm(test_loader, total=len(test_loader))

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(voader):
                if i >= 32:
                    voader.close()
                    break
                img, labels = data
                img = img.to(dev)
                labels = labels.to(dev)
                y, recon = model(img, labels)
                if i == 0:
                    imshow(img, labels, path, 'real')
                    imshow(recon, labels, path, 'reconstructed')
                _, pred = torch.max(y.data, 1)
                # print(pred)
                # print(labels)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        voader.close()
        acc.append(correct/total)
        plot_show(terror, tmean, acc, path)
        print(f'\n{correct}, {total}, accuracy: {100 * (correct/total)}%')

        torch.save(
            {
                'model': model.state_dict(),
                'epoch': e
            },
            'capsule.pth'
        )


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--mode',
                      action='store',
                      dest='mode',
                      default='mono',
                      help='Nature of dataset: mono/rgb')
    args.add_argument('--image_h',
                      action='store',
                      dest='ih',
                      default='64',
                      help='height of image')
    args.add_argument('--image_w',
                      action='store',
                      dest='iw',
                      default='64',
                      help='width of image')
    args.add_argument('--num_caps',
                      action='store',
                      dest='ncaps',
                      default='8',
                      help='number of primary capsules')
    args.add_argument('--num_classes',
                      action='store',
                      dest='nclass',
                      default='10',
                      help='number of classification classes')
    args.add_argument('--num_in_channels',
                      action='store',
                      dest='icchannels',
                      default='256',
                      help='number of output channels for the initial conv')
    args.add_argument('--num_po_channels',
                      action='store',
                      dest='ncoc',
                      default='32',
                      help='number of output capsules from the primary caps')
    args.add_argument('--num_do_channels',
                      action='store',
                      dest='ndoc',
                      default='16',
                      help='The output vector length for the digit caps')
    args.add_argument('--use_padding',
                      action='store',
                      dest='usepad',
                      default='False',
                      help='boolean: should the primary caps use padding or not')
    args.add_argument('--batch_size',
                      action='store',
                      dest='bsize',
                      default='8',
                      help='Batch size')
    args.add_argument('--data_dir',
                      action='store',
                      dest='datadir',
                      default='data',
                      help='path to dataset')

    parsed_args = args.parse_args()

    model = CapsuleNetwork(
        img_size=(int(parsed_args.ih), int(parsed_args.ih)),
        num_pcaps=int(parsed_args.ncaps),
        num_classes=int(parsed_args.nclass),
        ic_channels=int(parsed_args.icchannels),
        num_coc=int(parsed_args.ncoc),
        num_doc=int(parsed_args.ndoc),
        mode=parsed_args.mode,
        use_padding=True if parsed_args.usepad == 'True' else False
    )
    train_loader, test_loader = get_dataset(parsed_args.datadir, int(parsed_args.bsize), int(parsed_args.ih), int(parsed_args.iw))
    train(model, train_loader, test_loader, parsed_args.datadir)




