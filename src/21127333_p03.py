import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import platform
import matplotlib.pyplot as plt

# Print system information
print("System Information:")
print(f"  Operating System: {platform.system()} {platform.release()}")
print(f"  Processor: {platform.processor()}")

# Print device information
if torch.cuda.is_available():
    print("CUDA (GPU) is available!")
    print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("CUDA (GPU) is not available. Using CPU.")

# Print newline for separation
print()

class Discriminator(nn.Module):
    def __init__(self):
        """
        Discriminator model for distinguishing between real and fake images.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input size: 28x28 (MNIST image), Output size: 128
        self.nonlin1 = nn.LeakyReLU(0.2)  # Leaky ReLU activation function with slope 0.2
        self.fc2 = nn.Linear(128, 1)  # Output size: 1 (Binary classification: real or fake)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function to output probability

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor representing an image.

        Returns:
            torch.Tensor: Output tensor representing the probability of the input being real.
        """
        x = x.view(x.size(0), -1)  # Flatten the input image tensor
        h = self.nonlin1(self.fc1(x))  # Pass through first fully connected layer and activation function
        out = self.fc2(h)  # Output layer without activation function
        out = self.sigmoid(out)  # Apply sigmoid activation to obtain probability
        return out

class Generator(nn.Module):
    def __init__(self):
        """
        Generator model for generating fake images.
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)  # Input size: 100 (latent space), Output size: 128
        self.nonlin1 = nn.LeakyReLU(0.2)  # Leaky ReLU activation function with slope 0.2
        self.fc2 = nn.Linear(128, 28*28)  # Output size: 28x28 (MNIST image)

    def forward(self, x):
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Input tensor representing a noise vector.

        Returns:
            torch.Tensor: Output tensor representing a generated image.
        """
        h = self.nonlin1(self.fc1(x))  # Pass through first fully connected layer and activation function
        out = self.fc2(h)  # Output layer without activation function
        out = torch.tanh(out)  # Apply hyperbolic tangent activation to map output to range [-1, 1]
        out = out.view(out.size(0), 1, 28, 28)  # Reshape output to image dimensions
        return out

    
location_path='/MNIST'
dataset = torchvision.datasets.MNIST (root=location_path,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.5,), (0.5,))]),
                                      download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
# Re-initialize D, G:
D = Discriminator().to(device)
G = Generator().to(device)

# Now let's set up the optimizers (Adam, better than SGD for this)
optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
# optimizerD = torch.optim.Adam(D. parameters(), lr=0.0002)
# optimizerG = torch.optim.Adam = (G.parameters(), lr=0.0002)

criterion = nn.BCELoss()
batch_size = 64
lab_real = torch.ones(batch_size, 1)
lab_fake = torch.zeros(batch_size, 1)

num_epochs = 20

# Open a file to write logs
with open('gan_logs.txt', 'w') as f:
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # STEP 1: Discriminator optimization step
            x_real, _ = data
            x_real = x_real.to(device)
            lab_real = torch.ones(x_real.size(0), 1, device=device)
            optimizerD.zero_grad()
            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real)
            z = torch.randn(x_real.size(0), 100, device=device)
            x_gen = G(z).detach()
            lab_fake = torch.zeros(x_real.size(0), 1, device=device)
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # STEP 2: Generator optimization step
            optimizerG.zero_grad()
            z = torch.randn(x_real.size(0), 100, device=device)
            x_gen = G(z)
            lab_real = torch.ones(x_real.size(0), 1, device=device)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)
            lossG.backward()
            optimizerG.step()

            # Print the noise vector and loss to the file
            if i == 0 and epoch == 0:  # Print only once at the start of training
                f.write("Sample noise vector (first batch):\n")
                np.savetxt(f, z.cpu().numpy(), delimiter=",")  # Convert tensor to numpy array and save to file
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(dataloader)}], Discriminator Loss: {lossD.item()}, Generator Loss: {lossG.item()}\n")

# Generate images using the trained generator
num_samples_per_digit = 5
num_rows = 2
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5))

for i in range(10):  # Iterate over digits 0 to 9
    noise = torch.randn(num_samples_per_digit, 100)
    generated_images = G(noise).detach().numpy()

    for j in range(num_samples_per_digit):
        if i < 5:
            row = 0
        else:
            row = 1
            i -= 5
        ax = axes[row, i]
        ax.imshow(generated_images[j].reshape(28, 28), cmap='gray')
        ax.axis('off')

plt.tight_layout()
plt.savefig('test.png')  # Save the figure as 'test.png'
plt.show()
