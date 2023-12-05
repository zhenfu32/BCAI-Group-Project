import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import MLP, setup_model
from preprocessing import preprocess_data, convert_to_loader


def train_and_visualize(model, train_loader, val_loader, num_epochs, early_stop_threshold, criterion, optimizer):
    """
    Trains the model and visualizes the training process.

    :param model: The neural network model to be trained.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of epochs for training.
    :param early_stop_threshold: Early stopping threshold.
    :param criterion: Loss criterion.
    :param optimizer: Optimizer.
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
        if no_improve_epoch == early_stop_threshold:
            print("Early stopping triggered at epoch", epoch + 1)
            break

    # Plotting
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Losses')
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader, criterion):
    """
    Evaluates the model on the test set.

    :param model: Trained neural network model.
    :param test_loader: DataLoader for test data.
    :param criterion: Loss criterion.
    :return: Test loss.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    return test_loss / len(test_loader)

class ModelTrainingPipeline:
    def __init__(self, train_data, valid_data, test_data, hidden_layers, dropout_rate, seed, batch_size, num_epochs, early_stop_threshold, save_parameters=True, parameter_path="./model_parameters/", model_name="MNS_data_NN1"):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stop_threshold = early_stop_threshold
        self.save_parameters = save_parameters
        self.parameter_path = parameter_path
        self.model_name = model_name

        self._set_seed()
        self.train_loader = convert_to_loader(train_data, batch_size, shuffle=True)
        self.val_loader = convert_to_loader(valid_data, batch_size, shuffle=False)
        self.test_loader = convert_to_loader(test_data, batch_size, shuffle=False)

        self.model = setup_model(train_data, hidden_layers, dropout_rate)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def run(self):
        train_and_visualize(self.model, self.train_loader, self.val_loader, self.num_epochs, self.early_stop_threshold, self.criterion, self.optimizer)

        if self.save_parameters:
            torch.save(self.model.state_dict(), self.parameter_path + self.model_name + ".pth")
            print(f"Model parameters saved at {self.parameter_path + self.model_name + '.pth'}")

        test_loss = evaluate_model(self.model, self.test_loader, self.criterion)
        print(f'Test Loss: {test_loss:.4f}')




if __name__ == "__main__":
    data_path = './data/MyNewsScan_data.csv'
    train_data, valid_data, test_data = preprocess_data(data_path, standardization=True, creating_masks=False,splitting=(0.8, 0.1, 0.1))
    pipeline = ModelTrainingPipeline(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        hidden_layers=[64, 128, 128, 64],
        dropout_rate=0.5,
        seed=101,
        batch_size=16,
        num_epochs=100,
        early_stop_threshold=5,
        save_parameters=True,
        parameter_path="./model_parameters/",
        model_name="MNS_data_NN1"
    )

    pipeline.run()




