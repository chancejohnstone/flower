"""Generate server for fedht baseline."""

import sys
from collections import OrderedDict
from typing import Dict

import torch
import numpy as np
from torchvision import models
from fediot.model import train, test, init_batchnorm

# send fit round for history
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def gen_evaluate_fn(testloader, cfg, device: torch.device):
    """Get evaluate function for centralized metrics."""

    # global evaluation
    def evaluate(server_round, parameters, config):  # type: ignore
        """Define evaluate function for centralized metrics."""

        # define model
        num_classes = cfg.num_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights='IMAGENET1K_V1', progress=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        # set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate
