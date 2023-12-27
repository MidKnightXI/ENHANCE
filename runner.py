import torch
from torch import nn
from torchvision.utils import save_image
from os import path
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0),
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0),
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding = 0))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = (3,3), stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size = (3,3), stride = 1, padding = 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = DenoisingAutoencoder()
model.load_state_dict(torch.load('denoising_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

image_path = '/Users/midknight/Downloads/Why+are+my+photos+grainy+3.jpg'
output_path = path.splitext(image_path)[0] + '_denoised.jpg'

sample_image = Image.open(image_path).convert("RGB")

# Preprocess the image
input_image = transform(sample_image).unsqueeze(0)  # Add batch dimension

# Denoise the image using the trained model
with torch.no_grad():
    denoised_image = model(input_image)

save_image(denoised_image, output_path)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(sample_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image')
plt.imshow(denoised_image.squeeze(0).permute(1, 2, 0).numpy())
plt.axis('off')

plt.show()
