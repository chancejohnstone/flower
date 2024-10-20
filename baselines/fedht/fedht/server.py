"""Generate server for fedht baseline."""

from collections import OrderedDict
from typing import Dict

import torch

from fedht.model import test


# send fit round for history
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(testloader, model):
    """Get evaluate function for centralized metrics."""

    # global evaluation
    def evaluate(server_round, parameters, config):  # type: ignore
        """Define evaluate function for centralized metrics."""
        # set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader)
        return loss, {"accuracy": accuracy}

    return evaluate
