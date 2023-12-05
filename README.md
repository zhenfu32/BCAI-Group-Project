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

To train the models, execute the `main.py` file with the appropriate parameters. For example:

```bash
python main.py --model_name Baseline --num_epochs 100 --hidden_layers 128 128
```

Refer to the following optimal model parameters obtained from our experiments:



*Note: Replace `/path/to/image` with the actual path to the image showing the optimal model parameters.*

### Hyperparameter Tuning

We also provide a script for hyperparameter tuning using the Hyperband algorithm. Check the `modelTuning.py` file. Ensure you allocate sufficient virtual memory if you wish to use this script as it is resource-intensive.

### Contributions

If you wish to contribute to this project, please consider the following:

- Data preprocessing scripts can be enhanced for better feature engineering.
- Experiment with different neural network architectures and report the findings.
- Extend the interpretability analysis with additional or alternative methods.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Our team members who have contributed to this project in various capacities.
- The participants whose data formed the basis of our research insights.

---

*Please note that this README is a template and should be customized with the actual details of the project repository and file structure.*
