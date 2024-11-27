"""Run main for fedht baseline."""

import pickle
import torch

import flwr as fl
import hydra
import numpy as np
from flwr.common import NDArrays, ndarrays_to_parameters
from flwr.server.strategy.strategy import Strategy
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fediot.client import generate_client_fn_iot
from fediot.server import fit_round, gen_evaluate_fn
from fediot.utils import MyDataset


@hydra.main(config_path="conf", config_name="base_iot", version_base=None)
def main(cfg: DictConfig):
    """Run main file for fedht baseline.

    Parameters
    ----------
    cfg : DictConfig
        Config file for federated baseline; read from fedht/conf.
    """

    # set device to cuda:0, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load test image data
    num_features = cfg.num_features
    num_classes = cfg.num_classes

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),            # Resize image to 224x224 (typical for models like ResNet, VGG)
        transforms.ToTensor(),                   # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

     # import entire image dataset
    # dataset_path = 'C:/Users/CJOHNSTONE1/Documents/FL/HAR/supervised'
    dataset_path = 'C:/Users/CJOHNSTONE1/Documents/FL/HAR/supervised'
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


    # set client function
    client_fn = generate_client_fn_iot(
        cfg=cfg
    )

    # define strategy: fedavg
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=cfg.strategy.min_available_clients,
        evaluate_fn=gen_evaluate_fn(dataloader, cfg, device),
        on_fit_config_fn=fit_round
    )

    # start simulation
    np.random.seed(2025)
    hist_mnist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
    )

    if cfg.iterht:
        iterstr = "iter"
    else:
        iterstr = ""

    filename = (
        cfg.data
        + "_"
        + cfg.agg
        + iterstr
        + "_local"
        + str(cfg.num_local_epochs)
        + "_lr"
        + str(cfg.learning_rate)
        + "_numkeep"
        + str(cfg.num_keep)
        + ".pkl"
    )

    with open(filename, "wb") as file:
        pickle.dump(hist_mnist, file)


if __name__ == "__main__":
    main()
