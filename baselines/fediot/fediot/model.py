"""Define model for fedht baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import copy

def train(model, dataloader, cfg: DictConfig, device = torch.device):
    """Train model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs=cfg.num_local_epochs

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # model.train()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Scheduler step after each epoch
        # scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')    


def test(model, testloader: DataLoader, device: torch.device):
    """Test model."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    # initialize
    correct, total, loss = 0, 0, 0.0

    # put into evlauate mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs.float())
            total += labels.size(0)
            loss += criterion(outputs, labels.long()).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    loss /= len(testloader)
    accuracy = correct / total

    return loss, accuracy

def init_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        # Initialize the batch normalization layer parameters
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
