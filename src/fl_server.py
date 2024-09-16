from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
from flwr.common.typing import Parameters, Scalar
import mlflow

# List to store the loss values
loss_values = []

# Function to update the plot
def update_plot(loss_values):
    with open('/tmp/loss.txt', 'w') as file:
        file.write(','.join(f"{number:.2f}" for number in loss_values))
        

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        # Call the parent class method to get the aggregated loss
        aggregated_loss = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_loss is not None:
            loss_values.append(aggregated_loss[0])
            update_plot(loss_values)
        return aggregated_loss
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
