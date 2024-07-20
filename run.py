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

def parse_json_string(json_string):
    try:
        data = json.loads(json_string)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return data
        else:
            raise ArgumentTypeError("JSON string must be an array of strings.")
    except json.JSONDecodeError as e:
        raise ArgumentTypeError(f"Invalid JSON string: {e}")

def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser(
        prog="enhance",
        description="Deblur images using a lightweight ai model",
        epilog="If you encounter any problem please submit an issue here: https://github.com/MidKnightXI/ENHANCE")

    parser.add_argument("target",
                        type=parse_json_string,
                        required=True,
                        help="Define the JSON string specifying which directories the model will analyze the images from")
    parser.add_argument("-o", "--output",
                        type=str,
                        required=True,
                        help="Define the path of the output file eg: /out")
    args = parser.parse_args()
    return args

def denoise_image(model, image_path, output_directory):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    filename = os.path.basename(image_path)
    output_path = os.path.join(
        output_directory,
        f"{os.path.splitext(filename)[0]}_denoised.jpg")

    sample_image = Image.open(image_path).convert("RGB")

    input_image = transform(sample_image).unsqueeze(0)

    with torch.no_grad():
        denoised_image = model(input_image)

    save_image(denoised_image, output_path)

def denoise_images_in_directory(model, input_directory, output_directory):
    files = os.listdir(input_directory)
    os.makedirs(output_directory, exist_ok=True)

    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            denoise_image(model, image_path, output_directory)

def main():
    args = setup_argparse()

    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load('denoising_model.pth'))
    model.eval()

    targets = args.targets
    output_directory = args.output

    for target in targets:
        if os.path.isdir(target):
            denoise_images_in_directory(model, target, output_directory)
        elif os.path.isfile(target) and target.lower().endswith(('.jpg', '.jpeg', '.png')):
            denoise_image(model, target, output_directory)
        else:
            print(f"Invalid input path: {target}. Please provide a valid file or directory.")

if __name__ == "__main__":
    main()
