import polars as pl
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import os
import json
from datetime import date
import re
from dateutil import parser


class SalesForecaster:
    def __init__(self, data_path: str, models_base_dir: str = "prophet_models"):
        """
        Initializes the SalesForecaster.

        Args:
            data_path (str): Path to the sales data (e.g., sales_rt.parquet).
            models_base_dir (str): Directory to save/load trained Prophet models.
        """
        self.data_path = data_path
        self.pl_df = None  # Loaded on demand or if not already loaded
        self.models_base_dir = models_base_dir
        if not os.path.exists(self.models_base_dir):
            os.makedirs(self.models_base_dir)
        self.loaded_models = {}  # Cache for loaded Prophet models {model_key: model_object}

    def _load_and_prepare_base_data(self):
        """Loads and performs initial preparation of the base Polars DataFrame."""
        if self.pl_df is None:
            try:
                temp_df = pl.read_parquet(self.data_path)
                if "date" in temp_df.columns and temp_df["date"].dtype != pl.Date:
                    temp_df = temp_df.with_columns(
                        pl.col("date").cast(pl.Date, strict=False)
                    )
                self.pl_df = temp_df
            except Exception as e:
                print(f"Error loading base data in SalesForecaster: {e}")
                raise

    def _get_model_key(self, brand: str, region: str, target_metric: str) -> str:
        """Generates a unique key for a model based on brand, region, and metric."""
        return f"{brand.replace(' ', '_')}_{region.replace(' ', '_')}_{target_metric}".lower()

    def _get_model_path(self, model_key: str) -> str:
        """Generates the file path for a serialized model."""
        return os.path.join(self.models_base_dir, f"{model_key}.json")

    def _preprocess_for_prophet(
        self, brand: str, region: str, target_metric: str = "quantity"
    ) -> pd.DataFrame | None:
        """
        Filters and preprocesses data for a specific brand and region for Prophet.
        """
        if self.pl_df is None:
            self._load_and_prepare_base_data()

        if self.pl_df is None or not all(
            col in self.pl_df.columns
            for col in ["brand", "branch", "date", target_metric]
        ):
            print(
                f"SalesForecaster: Missing required columns (brand, branch, date, {target_metric}) in base DataFrame."
            )
            return None

        # Filter for the specific brand and region
        filtered_pl_df = self.pl_df.filter(
            (pl.col("brand") == brand) & (pl.col("branch") == region)
        )

        if filtered_pl_df.is_empty():
            print(
                f"SalesForecaster: No data found for brand '{brand}' in region '{region}'."
            )
            return None

        # Aggregate daily sums for the target metric
        daily_data_pl = (
            filtered_pl_df.group_by("date")
            .agg(pl.col(target_metric).sum().alias("y"))
            .sort("date")
        )

        if (
            daily_data_pl.is_empty() or daily_data_pl.height < 10
        ):  # Prophet needs some data
            print(
                f"SalesForecaster: Insufficient daily data for brand '{brand}', region '{region}' after aggregation."
            )
            return None

        # Rename 'date' to 'ds' for Prophet
        daily_data_pd = daily_data_pl.rename({"date": "ds"}).to_pandas()
        daily_data_pd["ds"] = pd.to_datetime(daily_data_pd["ds"])

        if (
            daily_data_pd.empty or len(daily_data_pd) < 2
        ):  # Prophet needs at least 2 data points
            print(
                f"SalesForecaster: Insufficient data after preprocessing for brand '{brand}', region '{region}'."
            )
            return None

        return daily_data_pd

    def _train_prophet_model(
        self, data_for_prophet: pd.DataFrame, model_path: str
    ) -> Prophet | None:
        """Trains a Prophet model and saves it."""
        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
            )
            model.fit(data_for_prophet)

            # Save the model
            with open(model_path, "w") as fout:
                json.dump(model_to_json(model), fout)
            return model
        except Exception as e:
            print(f"SalesForecaster: Error training Prophet model: {e}")
            return None

    def get_or_train_model(
        self, brand: str, region: str, target_metric: str = "quantity"
    ) -> Prophet | None:
        """
        Retrieves a trained Prophet model, training it if necessary.
        """
        model_key = self._get_model_key(brand, region, target_metric)
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        model_path = self._get_model_path(model_key)
        if os.path.exists(model_path):
            try:
                with open(model_path, "r") as fin:
                    model = model_from_json(json.load(fin))
                self.loaded_models[model_key] = model
                return model
            except Exception as e:
                print(
                    f"SalesForecaster: Error loading model from {model_path}: {e}. Retraining."
                )

        # If model not found or failed to load, train a new one
        data_for_prophet = self._preprocess_for_prophet(brand, region, target_metric)
        if data_for_prophet is None or data_for_prophet.empty:
            return None

        model = self._train_prophet_model(data_for_prophet, model_path)
        if model:
            self.loaded_models[model_key] = model
        return model

    def forecast_sales(
        self,
        brand: str,
        region: str,
        target_date: date,
        target_metric: str = "quantity",
    ) -> tuple[float | None, str | None]:
        """
        Generates a sales forecast for a specific brand, region, and target date.
        """
        if self.pl_df is None:  # Ensure base data is loaded if not already
            try:
                self._load_and_prepare_base_data()
            except Exception as e:
                return None, f"Failed to load base sales data: {e}"

        model = self.get_or_train_model(brand, region, target_metric)
        if not model:
            return (
                None,
                f"Could not get or train a forecast model for brand '{brand}', region '{region}'.",
            )

        try:
            # Create a future DataFrame for the specific target_date
            future_pd = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})

            forecast_pd = model.predict(future_pd)

            if forecast_pd.empty:
                return None, f"Prediction for {target_date} resulted in empty forecast."

            # Extract the forecasted value ('yhat') for the target date
            predicted_value = forecast_pd.loc[0, "yhat"]

            # Prophet can predict negative values, clip to 0 for sales/quantity
            predicted_value = max(0, predicted_value)

            return float(predicted_value), None
        except Exception as e:
            return None, f"Error during prediction for {target_date}: {e}"
