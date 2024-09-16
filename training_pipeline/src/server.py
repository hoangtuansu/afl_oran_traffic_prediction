import flwr as fl
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
import os

# List to store the loss values
loss_values = []
experiment_name = "Federated-Learning-Traffic-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)


# Function to update the plot
def update_plot(loss_values):
    plt.clf()  # Clear the current figure
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss per Round')
       
    plt.grid(True)
    plt.savefig(f'loss_plot.png')  # Save plot as image

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        # Call the parent class method to get the aggregated loss
        aggregated_loss = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_loss is not None:
            loss_values.append(aggregated_loss[0])
            update_plot(loss_values)
        return aggregated_loss

# Initialize the plot
plt.ion()
plt.figure()

# Start the server
with mlflow.start_run(run_name=run_name) as mlflow_run:
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=CustomStrategy()
    )

# Keep the plot open after training is done
plt.ioff()
plt.show()
