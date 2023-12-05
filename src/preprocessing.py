import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

def preprocess_data(data_path, standardization=True, creating_masks=False, splitting=(0.8, 0.1, 0.1)):
    # Load the dataset
    mns_data = pd.read_csv(data_path)

    # Data Cleaning: Removing rows with any missing values
    mns_data_cleaned = mns_data.dropna()

    # Data Standardization
    if standardization:
        scaler = StandardScaler()
        mns_data_scaled = scaler.fit_transform(mns_data_cleaned)
    else:
        mns_data_scaled = mns_data_cleaned.values

    # Creating masks
    if creating_masks:
        num_features = mns_data_cleaned.shape[1]
        feature_indices = range(num_features)

        # Re-creating the dataset with target outputs
        expanded_data_with_target = []
        for index in feature_indices:
            mask = [1 if i == index else 0 for i in feature_indices]
            for row in mns_data_scaled:
                masked_row = [row[i] * (1 - mask[i]) for i in feature_indices]
                target_output = row[index]  # The target output is the value of the feature being predicted
                expanded_data_with_target.append(masked_row + mask + [target_output])

        # Convert the expanded data with target output into a DataFrame
        data_final = pd.DataFrame(expanded_data_with_target)
    else:
        data_final = pd.DataFrame(mns_data_scaled)

    # Data Splitting
    train_size, valid_size = splitting[0], splitting[0] + splitting[1]
    train_data, temp_data = train_test_split(data_final, train_size=train_size, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=valid_size/(valid_size + splitting[2]), random_state=42)

    return train_data, valid_data, test_data

def convert_to_loader(data, batch_size, shuffle=True):
    """
    Converts data from DataFrame to PyTorch DataLoader.

    :param data: DataFrame containing features and target.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: DataLoader for the given data.
    """
    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
if __name__ == "__main__":
    data_path = './data/MyNewsScan_data.csv'
    train_data, valid_data, test_data = preprocess_data(data_path, standardization=True, creating_masks=False, splitting=(0.8, 0.1, 0.1))
    print(test_data)

