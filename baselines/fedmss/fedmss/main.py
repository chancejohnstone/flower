"""Run main for fedmss baseline."""

import pickle
import random
import torch
import flwr as fl
import hydra

import numpy as np
import pandas as pd
from flwr.common import NDArrays, ndarrays_to_parameters
from flwr.server.strategy.strategy import Strategy
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split, KFold

from fedmss.client import generate_client_fn_ucihd
from fedmss.fedht import FedHT
from fedmss.server import fit_round, get_evaluate_fn
from fedmss.utils import create_log_reg_and_instantiate_parameters, get_model_parameters, load_data, exhaustive, round_int

@hydra.main(config_path="conf", config_name="base_ucihd", version_base=None)
def main(cfg: DictConfig):
    """Run main file for fedmss baseline.

    Parameters
    ----------
    cfg : DictConfig
        Config file for federated baseline; read from fedht/conf.
    """
    # set seed
    random.seed(2024)

    # this vs. setting in cfg; what is preferred?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.data == "ucihd":

        # load UCI-HD data
        num_features = 13
        num_classes = 2 # binary classification
        
        # fetch dataset 
        # heart_disease = fetch_ucirepo(id=45) 
        
        # # data (as pandas dataframes) 
        # Xall = heart_disease.data.features 
        # yall = heart_disease.data.targets 

        # mask = np.isnan(Xall).any(axis=1)

        # # Filter out rows with NaN values from both X and y
        # X = np.array(Xall[~mask])
        # y = np.array(yall[~mask])
        # y = y > 0

        # load UCI-HD dataset
        X, y, names = load_data()
        y = np.array(y)

        # Split the data into training and testing sets (80% train, 20% test)
        X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_dataset = X_test, y_test

        # partition train data into clients
        kf = KFold(n_splits=cfg.num_clients, shuffle=True, random_state=43)
        X_train = [None] * cfg.num_clients
        y_train = [None] * cfg.num_clients
        X_val = [None] * cfg.num_clients
        y_val = [None] * cfg.num_clients
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_all), 1):
            X_train[fold-1], X_val[fold-1] = np.array(X_train_all)[train_idx,:], np.array(X_train_all)[val_idx,:]
            y_train[fold-1], y_val[fold-1] = np.array(y_train_all)[train_idx,:], np.array(y_train_all)[val_idx,:]

        train_dataset = X_train, y_train, X_val, y_val

        # define model
        model = create_log_reg_and_instantiate_parameters(cfg)

        #initial fit from first client
        model.fit(X_train[0], y_train[0])
        ndarrays = get_model_parameters(model, cfg)
        global_model_init = ndarrays_to_parameters(ndarrays)

        # set client function
        client_fn = generate_client_fn_ucihd(
            train_dataset,
            num_features=cfg.num_features,
            num_classes=cfg.num_classes,
            model=model,
            cfg=cfg,
            device=device
        )

    # initialize global model to all zeros
    # weights = np.zeros((num_classes, num_features))
    # bias = np.zeros(num_classes)
    # init_params_arr: NDArrays = [weights, bias]
    # init_params = ndarrays_to_parameters(init_params_arr)

    # define strategy: fedht
    strategy_fedht = FedHT(
        min_available_clients=cfg.strategy.min_available_clients,
        num_keep=cfg.num_keep,
        evaluate_fn=get_evaluate_fn(test_dataset, model, cfg),
        on_fit_config_fn=fit_round,
        iterht=cfg.iterht,
        initial_parameters=global_model_init
    )

    # define strategy: fedavg
    strategy_fedavg = fl.server.strategy.FedAvg(
        min_available_clients=cfg.strategy.min_available_clients,
        evaluate_fn=get_evaluate_fn(test_dataset, model, device),
        on_fit_config_fn=fit_round,
        initial_parameters=global_model_init
    )

    strategy: Strategy
    if cfg.agg == "fedht":
        strategy = strategy_fedht
    elif cfg.agg == "fedavg":
        strategy = strategy_fedavg
    else:
        print("Must select either fedht or fedavg for the aggregation strategy.")

    # # start simulation
    random.seed(2025)
    hist_ucihd = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
    )

    # import final non-integer FL model
    model_file = 'model.pkl'
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    results = get_model_parameters(model, cfg)
    index = np.array(np.where(np.abs(results[0][0]) > 0)[0]).astype(int).flatten()

    combinations, loss, accuracy = exhaustive(test_dataset, model, index, cfg)
    top_ind = np.argmin(loss)
    top_model = combinations[top_ind]
    params_int = round_int(results, top_model, index)
    # print(params_int)

    int_index = np.array(np.where(np.abs(params_int[0][0]) > 0)[0]).astype(int).flatten()
    # print(int_index)
    print(np.column_stack((np.array(names)[int_index], np.array(params_int[0][0])[int_index])))

    # print(int_model_params)

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
        pickle.dump(hist_ucihd, file)


if __name__ == "__main__":
    main()
