import flwr as fl
from flwr.client import NumPyClient, ClientApp, start_client
import mlflow
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple
import constants
from datetime import datetime
import os
import cords_semantics.tags as cords_tags


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
experiment_name = "Federated-Learning-Traffic-Classification"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)

mlflow.autolog()

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


def load_data(df):
    # Split dataset into train, validation, and test sets
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    trainloaders = DataLoader(CustomDataset(df), batch_size=constants.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(CustomDataset(test_df), batch_size=constants.BATCH_SIZE, shuffle=False)

    return trainloaders, testloader


def train(net, trainloader, epochs: int, verbose=False):
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

def evaluate(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
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

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader, current_round):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = Net(input_size=next(iter(trainloader))[0].shape[1]).to(DEVICE)
        self.current_round = current_round

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        with mlflow.start_run(run_name=run_name) as mlflow_run:
            train(self.model, self.trainloader, epochs=1)
            loss = evaluate(self.model, self.testloader, self.criterion)
            mlflow.set_tag(cords_tags.CORDS_RUN, mlflow_run.info.run_id)
            mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "ANN")
            mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
            mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.criterion)
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.optimizer)
            mlflow.log_metric("loss", loss)
            mlflow.pytorch.log_model(self.model, f"Round_{self.current_round}_Client")

        self.current_round += 1
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}


    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        test_loss = evaluate(self.model, self.testloader)
        # Return a dictionary of metrics (optional), e.g., loss.
        metrics = {"loss": test_loss}
        return float(test_loss), len(self.testloader.dataset), metrics
    


trainloaders, testloader = load_data(pmhistory_data)


def client_fn(current_round):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(trainloaders, testloader, current_round).to_client()


app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    current_round = 0 
    start_client(
        server_address=f'{aggregator_url}:51000',
        client=client_fn(current_round=current_round),
    )

