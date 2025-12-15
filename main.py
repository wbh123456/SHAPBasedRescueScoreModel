from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from pipeline import ModelPipeline
from result import Result

temp_file = "results/temp_data.pkl"
dtx_file = "results/dtx_raw_results.pkl"

LOAD_RESULT = temp_file
# LOAD_RESULT = None
NUM_REPEAT = 0
# SAVE_LOCATION = temp_file
SAVE_LOCATION = None


def main():
    result = Result()

    if LOAD_RESULT:
        result.load_from_file(LOAD_RESULT)

    if NUM_REPEAT > 0:
        dataset_name = "dtx"
        unprocessed_X, y = DataLoader(name=dataset_name).get_data(extra_days=4)
        X = DataPreprocessor(unprocessed_X).preprocess(show_plot=False)
        print(y.value_counts())

        pipeline = ModelPipeline(X, y, batch_size=4, epochs=80)
        results_raw = pipeline.run_repeated_5_fold(NUM_REPEAT, is_run_1_fold=False)
        result.merge_result(results_raw)

    if SAVE_LOCATION:
        result.save_to_file(SAVE_LOCATION)

    result.analyze_results()

    return


if __name__ == "__main__":
    main()
