import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

# Define a class to load the MNIST dataset and do all the training and testing
class MNIST:
    def __init__(self, batch_size=100, epochs=5, lr=0.001, augment=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.augment = augment
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.train_loader, self.test_loader = self.mnist_load()
        self.model = self.mnist_model().to(self.device)
        
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.hflip_accuracies = []
        self.vflip_accuracies = []
        self.noise_accuracies = {"0.01": [], "0.1": [], "1": []}
        self.checkpoint_dir = './checkpoints'
        if self.augment:
            self.checkpoint_dir += '_augmented'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def mnist_load(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if self.augment:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def mnist_model(self):
        # VGG 11
        return torchvision.models.vgg11(num_classes=10)
    
    def mnist_loss(self):
        return torch.nn.CrossEntropyLoss()
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        for epoch in range(self.epochs):
            self.model.train()  # Ensure model is in training mode
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            for i, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.mnist_loss()(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.epochs, i+1, len(self.train_loader), loss.item()))
            
            self.train_losses.append(total_train_loss / len(self.train_loader))
            self.train_accuracies.append(100 * correct_train / total_train)
            self.save_model(epoch)
            self.test(epoch)
        
    def test(self, epoch, test_loader=None):
        if test_loader is None:
            test_loader = self.test_loader

        self.model.eval()  # Ensure model is in evaluation mode
        total_test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.mnist_loss()(outputs, labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        accuracy = 100 * correct_test / total_test
        if test_loader is self.test_loader:
            self.test_losses.append(total_test_loss / len(test_loader))
            self.test_accuracies.append(accuracy)
            print(f'Epoch [{epoch+1}/{self.epochs}], Test Loss: {self.test_losses[-1]:.4f}, Test Accuracy: {accuracy:.2f}%')
        return accuracy

    def load_model_and_test(self, epoch, test_loader=None):
        self.model.load_state_dict(torch.load(f'{self.checkpoint_dir}/model_epoch_{epoch}.ckpt'))
        return self.test(epoch, test_loader)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f'{self.checkpoint_dir}/model_epoch_{epoch}.ckpt')

    def plot_metrics(self):
        epochs_range = range(1, self.epochs + 1)
        
        plt.figure(figsize=(12, 10))
        
        # Plot Test Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy vs. Epochs')
        plt.legend()

        # Plot Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy vs. Epochs')
        plt.legend()

        # Plot Test Loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Test Loss vs. Epochs')
        plt.legend()

        # Plot Training Loss
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, self.train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_flip(self):
        epochs_range = range(1, self.epochs + 1)

        plt.figure(figsize=(10, 6))
        
        # Plot Comparison of Test Accuracies
        plt.plot(epochs_range, self.test_accuracies, label='Original Test Accuracy')
        plt.plot(epochs_range, self.hflip_accuracies, label='HFlip Test Accuracy')
        plt.plot(epochs_range, self.vflip_accuracies, label='VFlip Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Test Accuracies on Original and Flipped Datasets')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_gaussian_noise(self):
        epochs_range = range(1, self.epochs + 1)

        plt.figure(figsize=(10, 6))
        
        # Plot Comparison of Test Accuracies
        plt.plot(epochs_range, self.test_accuracies, label='Original Test Accuracy')
        plt.plot(epochs_range, self.noise_accuracies["0.01"], label='Noisy Test Accuracy (0.01)')
        plt.plot(epochs_range, self.noise_accuracies["0.1"], label='Noisy Test Accuracy (0.1)')
        plt.plot(epochs_range, self.noise_accuracies["1"], label='Noisy Test Accuracy (1)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Test Accuracies on Original and Noisy Datasets')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def test_with_flips(self):
        # Create flipped test datasets
        hflip_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        vflip_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        hflip_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=hflip_transform, download=True)
        vflip_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=vflip_transform, download=True)

        hflip_test_loader = torch.utils.data.DataLoader(dataset=hflip_test_dataset, batch_size=self.batch_size, shuffle=False)
        vflip_test_loader = torch.utils.data.DataLoader(dataset=vflip_test_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            # Load model and test on original, horizontally flipped, and vertically flipped datasets
            print(f'Testing on original test dataset for epoch {epoch}')
            original_accuracy = self.load_model_and_test(epoch, self.test_loader)
            print(f'Testing on horizontally flipped test dataset for epoch {epoch}')
            hflip_accuracy = self.load_model_and_test(epoch, hflip_test_loader)
            print(f'Testing on vertically flipped test dataset for epoch {epoch}')
            vflip_accuracy = self.load_model_and_test(epoch, vflip_test_loader)
            print(f'Original Test Accuracy: {original_accuracy:.2f}%, HFlip Test Accuracy: {hflip_accuracy:.2f}%, VFlip Test Accuracy: {vflip_accuracy:.2f}%')

            self.hflip_accuracies.append(hflip_accuracy)
            self.vflip_accuracies.append(vflip_accuracy)

    def test_with_gaussian_noise(self):
        # Create noisy test datasets
        noise_transforms = {
            "0.01": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + np.sqrt(0.01) * torch.randn_like(x)),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            "0.1": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + np.sqrt(0.1) * torch.randn_like(x)),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            "1": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + np.sqrt(1.0) * torch.randn_like(x)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        }
        noise_test_datatsets = {}
        noise_test_loaders = {}
        for noise_level, noise_transform in noise_transforms.items():
            noise_test_datatsets[noise_level] = torchvision.datasets.MNIST(root='./data', train=False, transform=noise_transform, download=True)
            noise_test_loaders[noise_level] = torch.utils.data.DataLoader(dataset=noise_test_datatsets[noise_level], batch_size=self.batch_size, shuffle=False)
        
        for epoch in range(self.epochs):
            # Load model and test on original and noisy datasets
            print(f'Testing on original test dataset for epoch {epoch}')
            original_accuracy = self.load_model_and_test(epoch, self.test_loader)
            noise_accuracies = {}
            for noise_level, noise_test_loader in noise_test_loaders.items():
                print(f'Testing on noisy test dataset with noise level {noise_level} for epoch {epoch}')
                noise_accuracy = self.load_model_and_test(epoch, noise_test_loader)
                noise_accuracies[noise_level] = noise_accuracy
            print(f'Original Test Accuracy: {original_accuracy:.2f}%, Noisy Test Accuracies: {noise_accuracies}')

            self.noise_accuracies["0.01"].append(noise_accuracies["0.01"])
            self.noise_accuracies["0.1"].append(noise_accuracies["0.1"])
            self.noise_accuracies["1"].append(noise_accuracies["1"])

def MNIST_original_train(mnist: MNIST):
    mnist.train()
    mnist.plot_metrics()

def MNIST_flip_test(mnist: MNIST):
    mnist.test_with_flips()
    mnist.plot_flip()

def MNIST_gaussian_noise_test(mnist: MNIST):
    mnist.test_with_gaussian_noise()
    mnist.plot_gaussian_noise()

# Example usage
if __name__ == '__main__':
    mnist = MNIST()
    # MNIST_original_train(mnist)
    # MNIST_flip_test(mnist)
    MNIST_gaussian_noise_test(mnist)