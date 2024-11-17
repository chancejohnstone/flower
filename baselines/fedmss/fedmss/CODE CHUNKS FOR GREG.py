# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:55:25 2024

@author: jacob
"""

###BINARIZE
#Binarize X
X_binarized_df = helper_functions.convert_continuous_df_to_binary_df(X, num_quantiles=6)
X_binarized_df

df_all = X_binarized_df.merge(y, left_index = True, right_index = True)


## Initial Declarations
x_col_names = X_binarized_df.columns.tolist()
y_col_name = 'y'

num_clients = 2
train_prop = 0.8
num_features = 41
max_iterations = 150
tolerance = 0.001
sparsity_level = 6
alpha = 0.1
max_iter = 3
strategy = 'fed_iterht'

## Partition Data
df_train_bin, X_train_bin, X_test_bin, y_train_bin, y_test_bin, X_train_list_bin, y_train_list_bin = helper_functions.partition_dataset(
    df_all, x_col_names, y_col_name, num_clients, train_prop, False)

del X_train_bin['index']
del y_train_bin['index']


###PSEUDO 5 FOLD
# Initialize your models
models_standardized = {
    'DNN': dnn,
    'Logistic Regression': logistic_model,
    'Sparse Logistic Regression': elastic,
    'Federated Logistic Regression': final_model
    # Add other models that require standardized data
}

models_non_standardized = {
    'Random Forest': forest
    # Add other models that don't require standardized data
}

models_bin = {
    'Federated Sparse Logistic Regression': final_model_fed_sparse,
    'Federated MSS': mss
    # Add other models that don't require standardized data
}

# Define custom evaluation metric functions
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

# Initialize the metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score,
    'False Positive Rate': false_positive_rate,
    'False Negative Rate': false_negative_rate,
}

# Initialize the results DataFrame
results_df = pd.DataFrame(columns=['Partition', 'Model'] + list(metrics.keys()))

# Define number of partitions
num_partitions = 5

# Convert keys to a list for the DataFrame index
index_list = list(models_standardized.keys() | models_non_standardized.keys() | models_bin.keys())

# Iterate through each partition
for partition_num in range(num_partitions):
    # Loop through standardized models
    for model_name, model in models_standardized.items():
        partition_name = f'Partition {partition_num + 1}'
        # Split the dataset into train and test partitions
        partition_size = len(df_stand_test) // num_partitions
        test_start_index = partition_num * partition_size
        test_end_index = test_start_index + partition_size
        X_test = df_stand_test.iloc[test_start_index:test_end_index, :-1]
        y_test = df_stand_test.iloc[test_start_index:test_end_index, -1]
        # Loop through metrics
        # Predict on the test data
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5)

        # Convert predictions to integers (0 or 1)
        y_pred = y_pred.astype(int)

        # Convert true labels to integers (0 or 1)
        y_true = y_test.astype(int)

        # Calculate evaluation metrics
        metrics_values = []
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_true, y_pred)
            metrics_values.append(metric_value)

        # Store results in the dataframe
        results_df = results_df.append(pd.DataFrame([[partition_name, model_name] + metrics_values], columns=['Partition', 'Model'] + list(metrics.keys())), ignore_index=True)


    # Loop through non-standardized models
    for model_name, model in models_non_standardized.items():
        partition_name = f'Partition {partition_num + 1}'
        # Split the dataset into train and test partitions
        partition_size = len(df_test) // num_partitions
        test_start_index = partition_num * partition_size
        test_end_index = test_start_index + partition_size
        X_test = df_test.iloc[test_start_index:test_end_index, :-1]
        y_test = df_test.iloc[test_start_index:test_end_index, -1]
        # Loop through metrics
        # Predict on the test data
        y_pred = model.predict(X_test)

        # Convert predictions to integers (0 or 1)
        y_pred = y_pred.astype(int)

        # Convert true labels to integers (0 or 1)
        y_true = y_test.astype(int)

        # Calculate evaluation metrics
        metrics_values = []
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_true, y_pred)
            metrics_values.append(metric_value)

        # Store results in the dataframe
        results_df = results_df.append(pd.DataFrame([[partition_name, model_name] + metrics_values], columns=['Partition', 'Model'] + list(metrics.keys())), ignore_index=True)


    for model_name, model in models_bin.items():
        partition_name = f'Partition {partition_num + 1}'
        # Split the dataset into train and test partitions
        partition_size = len(df_bin_test) // num_partitions
        test_start_index = partition_num * partition_size
        test_end_index = test_start_index + partition_size
        X_test = df_bin_test.iloc[test_start_index:test_end_index, :-1]
        y_test = df_bin_test.iloc[test_start_index:test_end_index, -1]
        # Loop through metrics
        # Predict on the test data
        y_pred = model.predict(X_test)

        # Convert predictions to integers (0 or 1)
        y_pred = y_pred.astype(int)

        # Convert true labels to integers (0 or 1)
        y_true = y_test.astype(int)

        # Calculate evaluation metrics
        metrics_values = []
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_true, y_pred)
            metrics_values.append(metric_value)

        # Store results in the dataframe
        results_df = results_df.append(pd.DataFrame([[partition_name, model_name] + metrics_values], columns=['Partition', 'Model'] + list(metrics.keys())), ignore_index=True)


# Display the results
print(results_df)


def format_mean_std(series):
    mean = series.mean()
    std = series.std()
    return f"{mean:.3f} ± {std:.3f}"

# Group by the 'Model' column and calculate the mean and standard deviation for each metric
grouped_results = results_df.groupby('Model').agg({'Accuracy': [format_mean_std],
                                                   'Precision': [format_mean_std],
                                                   'Recall': [format_mean_std],
                                                   'F1 Score': [format_mean_std],
                                                   'False Positive Rate': [format_mean_std],
                                                   'False Negative Rate': [format_mean_std]})

# Rename the columns
grouped_results.columns = ['_'.join(col).strip() for col in grouped_results.columns.values]

# Define the custom list of model names in the desired order
custom_model_order = ['DNN', 'Random Forest', 'Logistic Regression', 'Sparse Logistic Regression',
                      'Federated Logistic Regression', 'Federated Sparse Logistic Regression', 'Federated MSS']

# Sort the index of the grouped dataframe using the custom list
grouped_results_sorted = grouped_results.reindex(custom_model_order)
grouped_results_sorted.columns = grouped_results_sorted.columns.str.replace('_format_mean_std', '')

print(grouped_results_sorted)
