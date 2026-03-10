import argparse
import os

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from pipeline import ModelPipeline
from result import Result

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 60
DEFAULT_REPEATS = 20


TOP_FEATURES = [
    "Weighted Mean Firing Rate (Hz)",
    "Number of Bursts",
    "Mean ISI within Network Burst - Avg (sec)",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP-Based Rescue Score Model"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a pickle file to load previous results from",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save results to (e.g. results/temp_data.pkl)",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"Number of repeated 5-fold CV runs (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize top features (bypasses training, requires --load)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.visualize and not args.load:
        raise SystemExit(
            "Error: --visualize requires --load to specify a results file."
        )

    result = Result()

    if args.load:
        result.load_from_file(args.load)

    dataset_name = "dtx"
    unprocessed_X, y = DataLoader(name=dataset_name).get_data(extra_days=4)
    X = DataPreprocessor(unprocessed_X).preprocess(show_plot=False)
    print(y.value_counts())

    if args.visualize:
        result.visualize_top_features(TOP_FEATURES, unprocessed_X, y)
        return

    if args.repeats > 0:
        pipeline = ModelPipeline(
            X, y, batch_size=DEFAULT_BATCH_SIZE, epochs=args.epochs
        )
        results_raw = pipeline.run_repeated_5_fold(
            args.repeats, is_run_1_fold=False
        )
        result.merge_result(results_raw)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        result.save_to_file(args.save)

    result.analyze_results()


if __name__ == "__main__":
    main()
