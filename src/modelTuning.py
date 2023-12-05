import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from ray import tune
from tqdm import tqdm
from ray.tune.schedulers import HyperBandScheduler
from ray import train as ray_train

from models import MLP, setup_model
from preprocessing import preprocess_data, convert_to_loader

class ModelTrainingPipeline:
    def __init__(self, config, train_data, valid_data, test_data, parameter_path="./model_parameters/", model_name="MNS_data_NN1"):
        self.config = config
        self.parameter_path = parameter_path
        self.model_name = model_name

        self._set_seed(config['seed'])
        self.train_loader = convert_to_loader(train_data, config['batch_size'], shuffle=True)
        self.val_loader = convert_to_loader(valid_data, config['batch_size'], shuffle=False)
        self.test_loader = convert_to_loader(test_data, config['batch_size'], shuffle=False)

        self.model = setup_model(train_data, config['hidden_layers'], config['dropout_rate'])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_and_visualize(self):
        best_val_loss = float('inf')
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}"):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)

            ray_train.report({'loss': avg_val_loss})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
            if no_improve_epoch == self.config['early_stop_threshold']:
                print("Early stopping triggered at epoch", epoch + 1)
                break

    def run(self):
        self.train_and_visualize()
        if self.config.get("save_parameters", False):
            torch.save(self.model.state_dict(), self.parameter_path + self.model_name + ".pth")
            print(f"Model parameters saved at {self.parameter_path + self.model_name + '.pth'}")

def tune_pipeline(config):
    data_path = 'E:/projects/DL and ML/BCAI4_MNW/src/data/MyNewsScan_data.csv'
    train_data, valid_data, test_data = preprocess_data(data_path, standardization=True, creating_masks=False, splitting=(0.8, 0.1, 0.1))

    pipeline = ModelTrainingPipeline(
        config = config,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        parameter_path="./model_parameters/",
        model_name="MNS_data_NN1"
    )
    pipeline.run()

if __name__ == "__main__":
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16]),
        "num_epochs": 50,
        "early_stop_threshold": 5,
        "hidden_layers": tune.choice([[64, 128], [128, 256]]),
        "dropout_rate": tune.uniform(0.3, 0.7),
        "seed": 101,
        "save_parameters": False  # Set to False to avoid saving parameters for each tuning trial
    }

    scheduler = HyperBandScheduler(metric="loss", mode="min")

    analysis = tune.run(
        tune_pipeline,
        config=config,
        num_samples=20,
        scheduler=scheduler
        )

