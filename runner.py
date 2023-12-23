import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the denoising model
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = DenoisingCNN()
model.load_state_dict(torch.load('denoising_model.pth'))
model.eval()

# Define a transform for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load a sample image for denoising
image_path = 'path/to/your/sample/image.jpg'  # Replace with the path to your image
sample_image = Image.open(image_path).convert("RGB")

# Preprocess the image
input_image = transform(sample_image).unsqueeze(0)  # Add batch dimension

# Denoise the image using the trained model
with torch.no_grad():
    denoised_image = model(input_image)

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
