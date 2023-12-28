import os
import argparse
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

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

def setup_argparse():
    parser = argparse.ArgumentParser(description='Denoise images using a pre-trained autoencoder model.')
    parser.add_argument(
        'input_directory',
        type=str,
        help='Path to the input directory containing images for denoising.')
    return parser.parse_args()

def denoise_images_in_directory(model, input_directory, output_directory):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            output_path = os.path.join(
                output_directory,
                f"{os.path.splitext(filename)[0]}_denoised.jpg")

            sample_image = Image.open(image_path).convert("RGB")

            input_image = transform(sample_image).unsqueeze(0)

            with torch.no_grad():
                denoised_image = model(input_image)

            save_image(denoised_image, output_path)

def main():
    args = setup_argparse()

    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load('denoising_model.pth'))
    model.eval()

    input_directory = args.input_directory
    output_directory = os.path.join(input_directory, 'denoised_output')

    denoise_images_in_directory(model, input_directory, output_directory)

if __name__ == "__main__":
    main()
