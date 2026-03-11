import argparse
import os

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from pipeline import ModelPipeline
from result import Result
from visualization import visualize_feature_distributions

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 60
DEFAULT_REPEATS = 0


FEATURES_TO_VISUALIZE = {
    "dtx": [
        "Weighted Mean Firing Rate (Hz)",
        "Mean ISI within Network Burst - Avg (sec)",
        "Inter-Burst Interval - Avg (sec)",
        "Burst Peak (Max Spikes per sec)",
        "IBI Coefficient of Variation - Avg",
        "Network Burst Duration - Avg (sec)",
    ],
    "genetic_ko": [
        "Area Under Normalized Cross-Correlation",
        "mean ISI within Network Burst - Avg (sec)",
        "Mean ISI within Burst - Avg (sec)",
        "Full Width at Half Height of Normalized Cross-Correlation",
        "Network Burst Duration - Avg (sec)",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP-Based Rescue Score Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dtx",
        help="Dataset to use, this should be one of the keys in data_config.py",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a pickle file to load previous results from",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save results to (e.g. results/temp_data.pkl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeated 5-fold CV runs (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--visualize_features",
        action="store_true",
        help="Visualize top features (bypasses training, requires --load)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    unprocessed_X, y = DataLoader(name=args.dataset).get_data(extra_days=3)
    X = DataPreprocessor(unprocessed_X).preprocess(show_plot=False)
    print(y.value_counts())

    result = Result()

    if args.load:
        result.load_from_file(args.load)

    if args.visualize_features:
        visualize_feature_distributions(
            FEATURES_TO_VISUALIZE[args.dataset], unprocessed_X, y
        )
        return

    if not args.visualize_features and args.repeats > 0:
        pipeline = ModelPipeline(
            X, y, batch_size=DEFAULT_BATCH_SIZE, epochs=args.epochs
        )
        results_raw = pipeline.run_repeated_5_fold(args.repeats, is_run_1_fold=False)
        result.merge_result(results_raw)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        result.save_to_file(args.save)

    result.analyze_results()


if __name__ == "__main__":
    main()
