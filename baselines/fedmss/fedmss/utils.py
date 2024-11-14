"""Execute utility functions for fedht baseline."""
import math
import itertools
import warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np
from sklearn.linear_model import SGDClassifier
from flwr.common.typing import NDArrays
import pandas as pd
from sklearn.metrics import log_loss
import copy

def set_model_params(model: SGDClassifier, params: NDArrays, cfg) -> SGDClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def get_model_parameters(model: SGDClassifier, cfg) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_initial_params(model: SGDClassifier, cfg) -> None:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.arange(cfg.num_classes)

    model.coef_ = np.zeros((1, cfg.num_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))


def create_log_reg_and_instantiate_parameters(cfg):
    """Helper function to create a LogisticRegression model."""
    model = SGDClassifier(
        loss='log_loss',
        learning_rate='constant',
        tol=.001,
        eta0=cfg.learning_rate,
        max_iter=cfg.num_local_epochs,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
    )

    # # Setting initial parameters, akin to model.compile for keras models
    # set_initial_params(model, cfg)
    return model

def load_data():
    df = pd.read_csv('fedmss/UCI_HD.csv')
    # convert to array and save variable names
    data = df.to_numpy()
    data_headers = list(df.columns.values)
    N = data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = data[:, Y_col_idx]
    Y_name = [data_headers[j] for j in Y_col_idx]
    Y[Y == 0] = -1

    # setup X and X_names
    X_col_idx = [j for j in range(data.shape[1]) if j not in Y_col_idx]
    X = data[:, X_col_idx]
    X_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    # X = np.insert(arr = X, obj = 0, values = np.ones(N), axis = 1)
    # X_names.insert(0, '(Intercept)') # Don't change name or format, code is touchy

    return X, Y, X_names

def exhaustive(test_dataset, model, index, cfg):

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # test data
    X_test, y_test = test_dataset

    # get all possible combination patterns
    combinations = np.array(list(itertools.product([0, 1], repeat=cfg.num_keep)))

    loss = []
    accuracy = []
    params = get_model_parameters(model, cfg)
    model2 = create_log_reg_and_instantiate_parameters(cfg)

    for each in combinations:

        #perform rounding
        params2 = round_int(params, each, index)

        # save new params
        set_model_params(model, params2, cfg)

        # test model
        loss.append(log_loss(y_test, model.predict_proba(X_test)))
        accuracy.append(model.score(X_test, y_test))

    return combinations, loss, accuracy

def round_int(params, each, index):
    params2 = copy.deepcopy(params)
    up = np.array(np.where(each == 1)[0]).astype(int)
    down = np.array(np.where(each == 0)[0]).astype(int)
    params2[0][0][index[up]] = np.ceil(np.array(params[0][0])[index[up]])
    params2[0][0][index[down]] = np.floor(np.array(params[0][0])[index[down]])

    return params2


