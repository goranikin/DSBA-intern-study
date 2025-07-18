from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_path):
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    return train_dataset, test_dataset

def load_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    # for caching the dataset
    dataset_path = './datasets'
    train_dataset, test_dataset = load_dataset(dataset_path)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = load_dataloader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = load_dataloader(test_dataset, batch_size=100, shuffle=False)
