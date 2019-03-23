from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_dataset(path, batch_size, ih, iw):
    trf = transforms.Compose([ \
        transforms.Resize((ih,iw)),
        transforms.ToTensor()])

    train_set = datasets.STL10(path, 'train', transform=trf, download=True)
    test_set = datasets.STL10(path, 'test', transform=trf, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader