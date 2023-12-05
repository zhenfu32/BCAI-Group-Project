import argparse
from training import ModelTrainingPipeline, preprocess_data


def main(args):
    # Preprocess data
    data_path = args.data_path
    train_data, valid_data, test_data = preprocess_data(data_path, standardization=args.standardization,
                                                        creating_masks=args.masked, splitting=(0.8, 0.1, 0.1))

    # Initialize the training pipeline
    pipeline = ModelTrainingPipeline(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        hidden_layers=args.hidden_layers,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        early_stop_threshold=args.early_stop_threshold,
        save_parameters=args.save_parameters,
        parameter_path=args.parameter_path,
        model_name=args.model_name
    )

    # Run the training pipeline
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network model on the MyNewsScan dataset.')

    # Add arguments to the parser
    parser.add_argument('--data_path', type=str, default='./data/MyNewsScan_data.csv', help='Path to the data file.')
    parser.add_argument('--standardization', type=bool, default=False, help='Whether to standardize the data or not.')
    parser.add_argument('--masked', type=bool, default=False, help='Whether to masked the data or not.')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 128, 128, 64],
                        help='List of hidden layer sizes.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the neural network.')
    parser.add_argument('--seed', type=int, default=1001, help='Seed for random number generators.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--early_stop_threshold', type=int, default=5, help='Early stopping threshold.')
    parser.add_argument('--save_parameters', type=bool, default=True, help='Whether to save the model parameters.')
    parser.add_argument('--parameter_path', type=str, default="./model_parameters/",
                        help='Path to save the model parameters.')
    parser.add_argument('--model_name', type=str, default="MNS_data_NN1",
                        help='Name of the model for saving parameters.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
