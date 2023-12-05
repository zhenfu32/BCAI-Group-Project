# BCAI-Group-Project

# MyNewsScan Neural Network Project

This repository contains the implementation of three neural network models aimed at predicting the correctness of participants' answers based on their subjective views on articles and personal health and emotional status. These models include a baseline, masked, and personalized model, each serving to explore the relationship between various input features and the participants' understanding as reflected by their correctness scores.

## Getting Started

To get started with this project, you need to set up your environment and install the necessary dependencies.

### Prerequisites

- Python 3.9 environment
- PyTorch version 1.12.0 (installation instructions can be found on the [PyTorch official website](https://pytorch.org/))
- Other dependencies listed in `requirements.txt`

### Setup

1. **Create a new Python environment:**

    ```bash
    python -m venv mynewsscan-env
    ```

    Activate the environment:

    ```bash
    # On Windows
    mynewsscan-env\Scripts\activate

    # On Unix or MacOS
    source mynewsscan-env/bin/activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    For PyTorch, follow the official guide to ensure compatibility with your system's CUDA version (if applicable).

3. **Prepare Data:**

    Due to confidentiality and privacy, the data cannot be provided in this repository. Prepare two datasets named `MyNewsScan_data.csv` and `MNS_data_supplementary.csv`, which include additional features such as sleep hours, tiredness, excitement, motivation, depression, anxiety, and openness to experience. 

### Training the Models

To train the models, execute the `main.py` file with the appropriate hyperparameters. In addition, you can find the parameters for the three trained models that we provided under the model_parameters folder.


#### Baseline Model:
```bash
python main.py --model_name Baseline --data_path 'MyNewsScan_data.csv' --num_epochs 100 --dropout_rate 0.4 --batch_size 16 --hidden_layers 128 128 
```
#### Masked Model:
```bash
python main.py --model_name Masked --data_path 'MyNewsScan_data.csv' --masked True --num_epochs 50 --dropout_rate 0.4 --batch_size 128 --hidden_layers 256 
```
#### Personalized Model:
```bash
python main.py --model_name Personalized --data_path 'MNS_data_supplementary.csv' --num_epochs 50 --dropout_rate 0.5 --batch_size 64 --hidden_layers 128 64 
```

Refer to the following optimal model hyperparameters obtained from our experiments:

![image](Optimal model parameters.png)


### Hyperparameter Tuning

We also provide a script for hyperparameter tuning using the Hyperband algorithm. Check the `modelTuning.py` file. Ensure you allocate sufficient virtual memory if you wish to use this script as it is resource-intensive. 

### Contributions

If you wish to contribute to this project, please consider the following:

- Data preprocessing scripts can be enhanced for better feature engineering.
- Experiment with different neural network architectures and report the findings.
- Extend the interpretability analysis with additional or alternative methods.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Our team members who have contributed to this project in various capacities.
- The participants whose data formed the basis of our research insights.

