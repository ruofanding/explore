import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from vit import VIT  # Assuming your VIT implementation is saved in a file named vit.py
from torch.utils.tensorboard import SummaryWriter
import datetime


# Load and preprocess the EMNIST dataset
def load_emnist(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_vit(model, train_loader, test_loader, criterion, optimizer, device, writer, epochs=10):
    model.train()
    total_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            if step % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'{name}/weights', param, total_step)
                    if param.grad is not None:
                        writer.add_histogram(f'{name}/gradients', param.grad, total_step)
                        writer.add_scalar(f'{name}/gradient_norm', param.grad.norm(), total_step)


            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            classes = model.classifier(outputs[:, 0, :])
            loss = criterion(classes, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training Loss', loss, total_step)

            running_loss += loss.item()
            if step % 100 == 0 and step > 0:
                print(step, running_loss / step)
            total_step += 1




        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        # Evaluate the model after each epoch
        test_vit(model, test_loader, device)

# Testing function
def test_vit(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(model.classifier(outputs[:, 0, :]), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    print(f'Precision: {precision:.4f}')

# Main function
def main():
    # Hyperparameters
    img_size = 28
    patch_size = 4
    channel_size = 1
    emb_dim = 128
    num_heads = 8
    num_layers = 6
    ffn_dim = 1024
    dropout = 0.1
    num_classes = 27  # 26 letters + 1 for "all letters"
    batch_size = 512
    learning_rate = 3e-4
    epochs = 10

    # Device configuration
    device = torch.device('mps')

    # Load data
    train_loader, test_loader = load_emnist(batch_size)
    writer = SummaryWriter('runs/vit_emnist_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Initialize the model, loss function, and optimizer
    model = VIT(img_size, patch_size, emb_dim, channel_size, num_heads, num_layers, ffn_dim, dropout, 'none').to(device)
    model.classifier = nn.Linear(emb_dim, num_classes).to(device)  # Add classifier head
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_vit(model, train_loader, test_loader, criterion, optimizer, device, writer, epochs)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'vit_emnist.pth')

if __name__ == '__main__':
    main()
