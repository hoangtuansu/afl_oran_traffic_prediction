import flwr as fl
import mlflow
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple
import argparse
from datetime import datetime
import os
import cords_semantics.tags as cords_tags

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
experiment_name = "Federated-Learning-Traffic-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)
mlflow.set_tracking_uri("http://10.180.113.115:32256")  


# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.feature_columns = [col for col in dataframe.columns if col != 'targetTput' and col != 'measTimeStampRf']
        self.target_column = 'targetTput'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[self.feature_columns].values.astype(np.float32)
        target = row[self.target_column].astype(np.float32)
        return features, target

def load_csv_dataset(filepath):
    df = pd.read_csv(filepath)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'measTimeStampRf' in categorical_columns:
        categorical_columns.remove('measTimeStampRf')

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_features = ohe.fit_transform(df[categorical_columns])
    ohe_feature_names = ohe.get_feature_names_out(categorical_columns)

    ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)
    df.drop(columns=categorical_columns + ['measTimeStampRf'], inplace=True)
    df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    train_partitions = np.array_split(train_df, 10)  # Adjust NUM_CLIENTS as needed

    trainloaders = [DataLoader(CustomDataset(partition), batch_size=32, shuffle=True) for partition in train_partitions]
    testloader = DataLoader(CustomDataset(test_df), batch_size=32, shuffle=False)

    return trainloaders, testloader

def train(net, trainloader, epochs: int, criterion, optimizer, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(trainloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {running_loss}")

def test(net, testloader, criterion):
    """Evaluate the network on the entire test set."""
    total_loss = 0.0
    net.eval()
    with torch.no_grad():
        for features, targets in testloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()
    total_loss /= len(testloader.dataset)
    return total_loss

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,  cid, trainloader, testloader, current_round):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = Net(input_size=next(iter(trainloader))[0].shape[1]).to(DEVICE)
        self.current_round = current_round
        self.criterion = torch.nn.MSELoss()
        self.client_id = cid
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        with mlflow.start_run(run_name=f'{self.client_id}_{run_name}') as mlflow_run:
            train(self.model, self.trainloader, epochs=1, criterion=self.criterion, optimizer=self.optimizer)
            loss = test(self.model, self.testloader, self.criterion)           
            mlflow.set_tag(cords_tags.CORDS_RUN, mlflow_run.info.run_id)
            mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "NN")
            mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
            mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.criterion)
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.optimizer)
            mlflow.log_metric("loss", loss)
            mlflow.pytorch.log_model(self.model, f"Round_{self.current_round}_Client_{self.client_id}")
        
        self.current_round += 1
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}


    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        test_loss = test(self.model, self.testloader, self.criterion)
        # Return a dictionary of metrics (optional), e.g., loss.
        metrics = {"loss": test_loss}
        return float(test_loss), len(self.testloader.dataset), metrics

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        choices=[0, 1],
        default=0,
        type=int,
        help="Partition of the dataset divided into 2 iid partitions created artificially.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV data file.",
    )
    # parser.add_argument(
    #     "--target-column",
    #     type=str,
    #     required=True,
    #     help="Target column name in the CSV file.",
    # )
    #python3 client_new_1.py --client-id 0 --data-path src_ue.csv 
    args = parser.parse_args()
    "This is the data file use for all clients"
    trainloaders, testloader = load_csv_dataset(args.data_path)

    client_partition = args.client_id

    client_trainloader = trainloaders[client_partition]
    client_testloader = testloader

    current_round = 0 

    client = FlowerClient(args.client_id, client_trainloader, client_testloader, current_round).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)
