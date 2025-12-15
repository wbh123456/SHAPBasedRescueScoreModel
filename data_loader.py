import pandas as pd


from constants import DAY_OF_INTEREST, KEY_INDEXES, START_DAY, WTTrue
from data_config import data_config


class DataLoader:
    def __init__(self, name):
        self.raw_data = self._get_raw_data(name)
        self.name = name
        return

    # Load and Transform Data
    def _get_raw_data(self, name: str) -> pd.DataFrame:
        all_rounds = data_config[name]
        data_list = []

        for round_data in all_rounds:
            data = pd.read_csv(round_data["path"])
            data["round"] = round_data["round"]
            data_list.append(data)
        return pd.concat(data_list)

    def _append_day_col(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure 'date' column is datetime type
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # Step 1: Find min date for each round
        min_dates = df.groupby("round")["date"].min().rename("min_date")

        # Step 2: Merge min_date back into the main df
        df = df.merge(min_dates, on="round")

        # Step 3: Calculate 'day' column (days since min_date per round)
        df["day"] = (df["date"] - df["min_date"]).dt.days + START_DAY

        # drop 'min_date' if no longer needed
        df.drop(columns="min_date", inplace=True)

        # Step 4: Count how many unique days are in each round
        days_per_round = df.groupby("round")["day"].nunique().rename("num_days")

        # Optional: merge with df if needed
        # df = df.merge(data_per_day, on=['round', 'day'])

        # Display results
        print("Number of days per round:")
        print(days_per_round)

        return df

    # 1. Append day info
    # 2. pivot data
    # 3. Filter label
    def _transform(
        self, appendWellIndex: bool = False, extra_days: int = 0
    ) -> pd.DataFrame:
        # Append day info
        df = self._append_day_col(self.raw_data)

        # Pivot the table
        pivoted_df = df.pivot(index=KEY_INDEXES, columns="parameter", values="value")

        data = pivoted_df[:]
        data = data.reset_index()

        data["label"] = data["geno"] + data["treatment"].astype(str)

        # We do not need the WT Treated data
        data = data[data["label"] != WTTrue]

        # !!! Filter day according to day of interest
        data = data[data["day"] <= DAY_OF_INTEREST + extra_days]
        data = data[data["day"] >= DAY_OF_INTEREST - extra_days]

        # Count how many data points per day in each round
        data_per_day = (
            data.groupby(["round", "day"]).size().rename("data_count").reset_index()
        )

        print("\n- Number of data points per (round, day):")
        print(data_per_day)

        if appendWellIndex:
            data["wellIndex"] = (
                data["filename"]
                + data["geno"].astype(str)
                + data["treatment"].astype(str)
                + data["day"].astype(str)
                + data["bioDup"].astype(str)
                + data["techDup"].astype(str)
            )

        data.pop("filename")
        data.pop("well")
        data.pop("geno")
        data.pop("treatment")
        data.pop("bioDup")
        data.pop("techDup")

        return data

    def get_data(self, extra_days=0):
        """
        Load and transform data for model training or evaluation.

        Args:
            extra_days (int, optional): Number of additional days to include
                around the day of interest when filtering data. Defaults to 0.
                The data will be filtered to include days in the range
                [DAY_OF_INTEREST - extra_days, DAY_OF_INTEREST + extra_days].
        """
        print("===Loading and Transforming Data", self.name)
        x = self._transform(extra_days=extra_days)
        y = x.pop("label")

        print("Total sample size =", x.shape[0])
        print("Total feature size =", x.shape[1])

        return x, y
