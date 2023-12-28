import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sys import stdout
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image

class DenoisingDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_size=(263, 263)):
        self.data = pd.read_csv(csv_path, header=0, names=['path', 'label'])
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        image = Image.open(img_path).convert('RGB')

        image = image.resize(self.target_size, Image.BICUBIC)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def setup_device():
    device = torch.device("cpu")

    # if torch.cuda.is_available(): # Uncomment for Nvidia GPUs
    #     DEVICE = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    stdout.write(f"Training on {device}")
    stdout.flush()
    return device


def train_model(model, data_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['image'].to(device)
            targets = batch['image'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        stdout.write(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}\n')
        stdout.flush()

    model.to(torch.device("cpu"))
    torch.save(model.state_dict(), "denoising_model.pth")


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = DenoisingDataset(csv_path='./dataset/dataset_info.csv',
                                     transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    DEVICE = setup_device()
    EPOCH = 10

    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_model(model, data_loader, criterion, optimizer, DEVICE, EPOCH)


if __name__ == "__main__":
    main()
