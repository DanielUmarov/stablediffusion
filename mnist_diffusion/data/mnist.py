from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size: int = 128, train: bool = True, num_workers: int = 2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # map to [-1, 1]
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=train)
    return dl
