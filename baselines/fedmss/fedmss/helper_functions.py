import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np
from typing import *
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.metrics import classification_report, log_loss
from sklearn.linear_model import LogisticRegression



def partition_dataset(df, x_col_names, y_col_name,
                      num_clients, train_prop, standardize=True):
    # Split the dataset into features (X) and the target variable (y)
    X = df[x_col_names]
    y = df[y_col_name]

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_prop, random_state=42)

    # Initialize empty lists to store data for each client
    X_train_list = []
    y_train_list = []

    if standardize == True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_stand = pd.DataFrame(scaler.transform(X_train),columns = x_col_names).reset_index()
        X_test_stand = pd.DataFrame(scaler.transform(X_test),columns = x_col_names)
        y_train = y_train.reset_index()
    else:
        X_train_stand = X_train.reset_index()
        X_test_stand = X_test
        y_train = y_train.reset_index()

    # Shuffle the training data randomly
    df_train = X_train_stand.merge(y_train, left_index = True, right_index = True)

    # Divide the shuffled training data into subsets for each client
    subset_size = len(df_train) // num_clients

    for i in range(num_clients):
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        X_client_train = df_train[start_idx:end_idx][x_col_names]
        y_client_train = df_train[start_idx:end_idx][y_col_name]
        X_train_list.append(X_client_train)
        y_train_list.append(y_client_train)

    # Return the training and test sets for the entire dataset, and the training sets for each client
    return df_train, X_train_stand, X_test_stand, y_train, y_test, X_train_list, y_train_list


def create_sgd_classifiers(X_train_list, y_train_list, 
                           num_clients, alpha, max_iter, tolerance):
    
    # initialize logistic regression model for each client
    models = []
    
    for i in range(num_clients):
        lr = SGDClassifier(loss='log', alpha = alpha, max_iter=max_iter, tol=tolerance, warm_start = True)
        lr.fit(X_train_list[i], y_train_list[i])
        models.append(lr)
        
    return models


def federated_logistic_regression(X_train_list, y_train_list, sparsity_level, 
                                  num_features, max_iterations, num_clients, 
                                  models, tolerance, strategy, X_train, y_train):
    global_model = np.random.uniform(low=-2, high=2, size=(1, num_features))
    global_intercept = 0

    global_model_losses = []
    
    # Create final model for loss calculation
    final_model = SGDClassifier(loss='log', alpha=0.1, max_iter=1, tol=tolerance, warm_start=True)
    final_model.fit(X_train, y_train)

    final_model.coef_ = global_model  
    final_model.intercept_ = global_intercept

    # Calculate and store the loss on the global model
    global_model_loss = log_loss(y_train, final_model.predict_proba(X_train)[:, 1])
    global_model_losses.append(global_model_loss)

    if strategy == 'fed_ht':
        for round in range(max_iterations):
            print("Round:", round)
            
            global_model_old = global_model.copy()
            global_intercept_old = global_intercept
            
            # Sum up gradients across clients
            sum_gradients = np.zeros((1, num_features))
            sum_intercepts = 0
            
            for i in range(num_clients):
                # Update each client's model with the global model
                models[i].coef_ = global_model  
                models[i].intercept_ = global_intercept
                # Train the model on the local data
                models[i].fit(X_train_list[i], y_train_list[i])
                
                # Get the gradient of the model parameters
                client_grad = models[i].coef_
                client_intercept = models[i].intercept_
                
                sum_gradients += client_grad
                sum_intercepts += client_intercept
            
            # Average the gradients
            global_model = sum_gradients / num_clients
            global_intercept = sum_intercepts / num_clients
            
            # Perform a hard threshold to keep the top sparsity_level coefficients
            top_indices = np.argsort(np.abs(global_model))[0, -sparsity_level:]
            thresholded_global_model = np.zeros((1, num_features))
            thresholded_global_model[0, top_indices] = global_model[0, top_indices]
            global_model = thresholded_global_model

            # Check for convergence
            if np.linalg.norm(global_model - global_model_old) < tolerance:
                print("Convergence achieved after", round+1, "rounds.")
                break
        
            # Create final model for loss calculation
            final_model = SGDClassifier(loss='log', alpha=0.1, max_iter=1, tol=tolerance, warm_start=True)
            final_model.fit(X_train, y_train)
    
            final_model.coef_ = global_model  
            final_model.intercept_ = global_intercept
        
            # Calculate and store the loss on the global model
            global_model_loss = log_loss(y_train, final_model.predict_proba(X_train)[:, 1])
            global_model_losses.append(global_model_loss)
    
    elif strategy == 'fedavg':
        for round in range(max_iterations):
            print("Round:", round)
            
            global_model_old = global_model.copy()
            global_intercept_old = global_intercept
            
            # Sum up gradients across clients
            sum_gradients = np.zeros((1, num_features))
            sum_intercepts = 0
            
            for i in range(num_clients):
                # Update each client's model with the global model
                models[i].coef_ = global_model  
                models[i].intercept_ = global_intercept
                # Train the model on the local data
                models[i].fit(X_train_list[i], y_train_list[i])
                
                # Get the gradient of the model parameters
                client_grad = models[i].coef_
                client_intercept = models[i].intercept_
                
                sum_gradients += client_grad
                sum_intercepts += client_intercept
            
            # Average the gradients
            global_model = sum_gradients / num_clients
            global_intercept = sum_intercepts / num_clients

            # Check for convergence
            if np.linalg.norm(global_model - global_model_old) < tolerance:
                print("Convergence achieved after", round+1, "rounds.")
                break
        
            # Create final model for loss calculation
            final_model = SGDClassifier(loss='log', alpha=0.1, max_iter=1, tol=tolerance, warm_start=True)
            final_model.fit(X_train, y_train)
    
            final_model.coef_ = global_model  
            final_model.intercept_ = global_intercept
        
            # Calculate and store the loss on the global model
            global_model_loss = log_loss(y_train, final_model.predict_proba(X_train)[:, 1])
            global_model_losses.append(global_model_loss)    

    elif strategy == 'fed_iterht':
        for round in range(max_iterations):
            print("Round:", round)
            
            global_model_old = global_model.copy()
            global_intercept_old = global_intercept
            
            # Sum up gradients across clients
            sum_gradients = np.zeros((1, num_features))
            sum_intercepts = 0
            
            for i in range(num_clients):
                # Update each client's model with the global model
                models[i].coef_ = global_model  
                models[i].intercept_ = global_intercept
                # Train the model on the local data
                models[i].fit(X_train_list[i], y_train_list[i])
                
                # Get the gradient of the model parameters
                client_grad = models[i].coef_
                client_intercept = models[i].intercept_
                
                # Perform a hard threshold to keep the top sparsity_level coefficients
                top_indices_client = np.argsort(np.abs(client_grad))[0, -sparsity_level:]
                thresholded_client_model = np.zeros((1, num_features))
                thresholded_client_model[0, top_indices_client] = client_grad[0, top_indices_client]
                client_grad = thresholded_client_model
                
                sum_gradients += client_grad
                sum_intercepts += client_intercept
            
            # Average the gradients
            global_model = sum_gradients / num_clients
            global_intercept = sum_intercepts / num_clients
            
            # Perform a hard threshold to keep the top sparsity_level coefficients
            top_indices = np.argsort(np.abs(global_model))[0, -sparsity_level:]
            thresholded_global_model = np.zeros((1, num_features))
            thresholded_global_model[0, top_indices] = global_model[0, top_indices]
            global_model = thresholded_global_model
    
            # Check for convergence
            if np.linalg.norm(global_model - global_model_old) < tolerance:
                print("Convergence achieved after", round+1, "rounds.")
                break
        
            # Create final model for loss calculation
            final_model = SGDClassifier(loss='log', alpha=0.1, max_iter=1, tol=tolerance, warm_start=True)
            final_model.fit(X_train, y_train)
    
            final_model.coef_ = global_model  
            final_model.intercept_ = global_intercept
        
            # Calculate and store the loss on the global model
            global_model_loss = log_loss(y_train, final_model.predict_proba(X_train)[:, 1])
            global_model_losses.append(global_model_loss)

    return global_model, global_intercept, global_model_losses


def round_values(arr):
    # Create a copy of the input array
    rounded_arr = np.copy(arr)

    if arr.shape == (1, 112):
        for i in range(rounded_arr.shape[1]):
            if abs(rounded_arr[0, i]) < 1:
                rounded_arr[0, i] = np.sign(rounded_arr[0, i]) * 1
            else:
                rounded_arr[0, i] = np.round(rounded_arr[0, i])
    elif arr.shape == (1,):
        rounded_arr[0] = np.round(rounded_arr[0])
    else:
        raise ValueError("Unsupported input shape")

    return rounded_arr



def convert_continuous_df_to_binary_df(df, num_quantiles=25, get_featureIndex_to_groupIndex=False):
    """Convert a dataframe with continuous features to a dataframe with binary features by thresholding

    Parameters
    ----------
    df : pandas.DataFrame
        original dataframe where there are columns with continuous features
    num_quantiles : int, optional
        number of points we pick from quantile as thresholds if a column has too many unique values, by default 25
    get_featureIndex_to_groupIndex : bool, optional
        whether to return a numpy array that maps feature index to group index, by default False

    Returns
    -------
    binarized_df : pandas.DataFrame
        a new dataframe where each column only has 0/1 as the feature
    """

    colnames = df.columns
    n = len(df)
    print("Converting continuous features to binary features in the dataframe......")
    print(f"If a feature has more than {num_quantiles} unique values, we pick the thresholds by selecting {num_quantiles} quantile points. You can change the number of thresholds by passing another specified number: convert_continuous_df_to_binary_df(df, num_quantiles=50).")

    percentile_ticks = np.linspace(0, 100, num=num_quantiles + 1)[1:-1]

    binarized_dict = {}

    featureIndex_to_groupIndex = []
    for i in range(0, len(colnames)):
        uni = df[colnames[i]].unique()
        if len(uni) == 2:
            binarized_dict[colnames[i]] = np.asarray(df[colnames[i]], dtype=int)
            featureIndex_to_groupIndex.append(i)
            continue

        # Convert column values to numeric
        df[colnames[i]] = pd.to_numeric(df[colnames[i]], errors='coerce')

        thresholds = np.percentile(df[colnames[i]].dropna(), percentile_ticks)
        for threshold in thresholds:
            tmp_feature = np.zeros(n, dtype=int)
            tmp_name = colnames[i] + ">" + str(threshold)

            greater_than_threshold_indices = df[colnames[i]] > threshold
            tmp_feature[greater_than_threshold_indices] = 1

            binarized_dict[tmp_name] = tmp_feature
            featureIndex_to_groupIndex.append(i)

    binarized_df = pd.DataFrame(binarized_dict)
    print("Finish converting continuous features to binary features......")
    if get_featureIndex_to_groupIndex:
        return binarized_df, np.asarray(featureIndex_to_groupIndex, dtype=int)
    return binarized_df
