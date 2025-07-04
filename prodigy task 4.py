import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# Basic generator network (UNet-style simplified)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load MNIST as a proxy for edge-to-digit style translation
def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = dset.MNIST(root='./data', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    return loader

def train(generator, data_loader, device):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    generator.train()
    for epoch in range(1):
        for i, (img, _) in enumerate(data_loader):
            img = img.to(device)
            noisy = img + 0.5 * torch.randn_like(img)  # fake 'edge map' (input)
            noisy = torch.clamp(noisy, -1.0, 1.0)

            optimizer.zero_grad()
            output = generator(noisy)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

    return generator

def show_samples(generator, data_loader, device):
    generator.eval()
    with torch.no_grad():
        for img, _ in data_loader:
            img = img.to(device)
            noisy = img + 0.5 * torch.randn_like(img)
            noisy = torch.clamp(noisy, -1.0, 1.0)

            output = generator(noisy)

            # show input vs output
            samples = torch.cat((noisy[:4], output[:4], img[:4]), 0)
            samples = samples.cpu()
            grid = vutils.make_grid(samples, nrow=4, normalize=True)
            plt.figure(figsize=(10, 5))
            plt.title("Input (noisy) | Output (generated) | Ground Truth")
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.show()
            break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    data_loader = get_data_loader()

    trained_generator = train(generator, data_loader, device)
    show_samples(trained_generator, data_loader, device)
