"""Generate client for fedht baseline."""
import sys 

from collections import OrderedDict
from typing import cast

import torch
import copy
import numpy as np
from flwr.client import Client, NumPyClient
from flwr.common import Context
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

from fediot.model import test, train, init_batchnorm
from fediot.utils import MyDataset



# IoT client
class IoTClient(NumPyClient):
    """Define IoTClient class."""

    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        cfg: DictConfig,
        device
    ) -> None:
        """IoT client for experimentation."""
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.cfg = cfg
        self.device = device

    # get parameters from existing model
    def get_parameters(self, config):
        """Get parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Fit model."""
        # set model parameters
        # Apply the initialization to your model
        # self.model.apply(init_batchnorm)

        params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # train model
        self.model.train()

        # training for local epochs defined by config
        train(self.model, self.trainloader, self.cfg, self.device)

        # print(self.model.state_dict().keys())
        # sys.exit()

        return self.get_parameters(self.model), self.num_obs, {}

    def evaluate(self, parameters, config):
        """Evaluate model."""
        # set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # need to change from log_loss to torch.loss and change other metrics
        loss, accuracy = test(self.model, self.testloader, self.device)

        return loss, self.num_obs, {"accuracy": accuracy}


# client fn for input into simulation
def generate_client_fn_iot(
    cfg: DictConfig
):
    """Generate client function for simulated FL."""

    # def client_fn(cid: int):
    def client_fn(context: Context) -> Client:
        """Define client function for centralized metrics."""
        # Get node_config value to fetch partition_id
        partition_id = cast(int, context.node_config["partition-id"])

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),            # Resize image to 224x224 (typical for models like ResNet, VGG)
            transforms.ToTensor(),                   # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

        # import client image dataset
        dataset_path = 'C:/Users/CJOHNSTONE1/Documents/FL/HAR/client' + partition_id
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

        # create dataloader
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        # define model and set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights='IMAGENET1K_V1', progress=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        return IoTClient(
            dataloader, dataloader, model, len(dataloader.dataset), cfg, device
        ).to_client()

    return client_fn
