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
        "Number of Spikes per Network Burst - Avg",
        "Area Under Normalized Cross-Correlation",
        "Network Burst Duration - Avg (sec)",
        "Burst Peak (Max Spikes per sec)",
        "Burst Duration - Avg (sec)",
    ],
}


def get_features_to_visualize(dataset):
    if "dtx" in dataset:
        return FEATURES_TO_VISUALIZE["dtx"]
    elif "genetic_ko" in dataset:
        return FEATURES_TO_VISUALIZE["genetic_ko"]
    else:
        raise ValueError(f"Dataset {dataset} not found")


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
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Plot rescue score and SHAP convergence vs. number of repeats (requires --load)",
    )
    parser.add_argument(
        "--show_zeros_removed",
        action="store_true",
        help="Add a side-by-side subplot with zeros removed to feature distribution plots",
    )
    args = parser.parse_args()
    if args.convergence and not args.load:
        parser.error("--convergence requires --load to be specified")
    if args.visualize_features and not args.dataset:
        parser.error("--visualize_features requires --dataset to be specified")
    if args.repeats > 0 and args.visualize_features:
        parser.error(
            "--repeats cannot be used together with --visualize_features"
        )
    return args


def main():
    args = parse_args()

    result = Result()

    if args.load:
        result.load_from_file(args.load)

    if args.convergence:
        load_stem = (
            os.path.splitext(os.path.basename(args.load))[0]
            if args.load
            else args.dataset
        )
        result.analyze_convergence(output_dir=os.path.join("convergence", load_stem))
        return

    unprocessed_X, y = DataLoader(name=args.dataset).get_data(extra_days=3)
    X = DataPreprocessor(unprocessed_X).preprocess(show_plot=False)
    print(y.value_counts())

    if args.visualize_features:
        dunn_stats = None
        if args.load and result.results:
            dunn_stats = result.get_dunn_stats()

        visualize_feature_distributions(
            get_features_to_visualize(args.dataset),
            unprocessed_X,
            y,
            output_dir=os.path.join("visualization", args.dataset),
            show_zeros_removed=args.show_zeros_removed,
            dunn_stats=dunn_stats,
            dataset=args.dataset,
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

        # Determine output directory for plots based on the save path
        # e.g., results/genetic_ko_env_3_rounds.pkl -> results/genetic_ko_env_3_rounds/
        save_name = os.path.splitext(os.path.basename(args.save))[0]
        output_dir = os.path.join(os.path.dirname(args.save), save_name)
    else:
        output_dir = None

    rescue_scores_summary, shift_denom = result.analyze_results(
        output_dir=output_dir, dataset=args.dataset
    )
    print(rescue_scores_summary)

    if args.save and args.repeats > 0:
        save_name = os.path.splitext(os.path.basename(args.save))[0]
        convergence_dir = os.path.join("convergence", save_name)
        result.analyze_convergence(output_dir=convergence_dir)


if __name__ == "__main__":
    main()
