import pandas as pd
import polars as pl
from groq import Groq
import warnings
import hashlib
import re
from datetime import datetime, date, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import brute
from itertools import product
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from top_model_context import PerformanceAnalyzer
from region_model_context import CustomerAnalyzer
from cross_sell_model_context import CrossSellingAnalyzer
from utils_helpers_model_context import rename_cols
from region_normalizer import normalize_region_name, normalize_region_list

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter(
    "ignore", RuntimeWarning
)  # Suppress other potential runtime warnings from statsmodels

import os
from dotenv import load_dotenv

dotenv_path = "D:\\Projects\\zipped\\.env"
load_dotenv(dotenv_path=dotenv_path)


class SalesDataChatbot:
    def __init__(self, data_path: str):
        self.excel_path = data_path
        self.pl_df = pl.read_parquet(data_path)
        self.pl_df = rename_cols(self.pl_df)

        # Clean whitespace from brand and branch columns
        if "brand" in self.pl_df.columns:
            self.pl_df = self.pl_df.with_columns(
                pl.col("brand").str.strip_chars().alias("brand")
            )
        if "branch" in self.pl_df.columns:
            self.pl_df = self.pl_df.with_columns(
                pl.col("branch").str.strip_chars().alias("branch")
            )
            self.pl_df = self.pl_df.with_columns(
                pl.col("branch")
                .map_elements(normalize_region_name, return_dtype=pl.String)
                .alias("branch")
            )

        if "date" in self.pl_df.columns:
            if self.pl_df["date"].dtype != pl.Date:
                try:
                    self.pl_df = self.pl_df.with_columns(
                        pl.col("date").cast(pl.Date, strict=False)
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not cast 'date' column to pl.Date: {e}. Date filtering might be affected."
                    )
        else:
            print(
                "Critical Warning: 'date' column not found. Many features including forecasting and plotting will be severely limited or fail."
            )

        metric_mapping = {
            "sales_value": "value",
            "delivered_value": "value",
            "delivered_qty": "quantity",
        }
        current_cols = self.pl_df.columns
        for original_name, standard_name in metric_mapping.items():
            if original_name in current_cols and standard_name not in current_cols:
                self.pl_df = self.pl_df.rename({original_name: standard_name})

        if "value" not in self.pl_df.columns:
            print("Warning: 'value' column (for sales value) not found.")
        if "quantity" not in self.pl_df.columns:
            print(
                "Warning: 'quantity' column not found. Quantity-based forecasting/plotting will be affected."
            )
        if "brand" not in self.pl_df.columns:
            print(
                "Warning: 'brand' column not found. Brand-specific features will be affected."
            )

        self.known_brands = []
        if "brand" in self.pl_df.columns:
            try:
                self.known_brands = self.pl_df["brand"].unique().drop_nulls().to_list()
            except Exception:
                pass
        self.known_brands = [str(b) for b in self.known_brands if b is not None]

        self.known_regions = []
        if "branch" in self.pl_df.columns:
            try:
                self.known_regions = (
                    self.pl_df["branch"].unique().drop_nulls().to_list()
                )
            except Exception:
                pass
        self.known_regions = [str(r) for r in self.known_regions if r is not None]

        self.performance_analyzer = PerformanceAnalyzer(self.pl_df)
        self.customer_analyzer = CustomerAnalyzer(self.pl_df)
        self.cross_selling_analyzer = CrossSellingAnalyzer(
            self.pl_df,
            order_id_col="order_id",
            item_id_col="sku_code",
            item_name_col="sku_name",
        )

        all_texts = (
            self.performance_analyzer.summaries
            + self.customer_analyzer.summaries
            + self.cross_selling_analyzer.summaries
            + self.cross_selling_analyzer.rules_text
        )
        self.full_context_text = "\n\n".join(filter(None, all_texts))

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.llm = Groq(api_key=groq_api_key)

    def _parse_date(self, date_str: str | None) -> date | None:
        if not date_str:
            return None
        formats_to_try = ["%Y-%m-%d"]
        formats_to_try.extend(["%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"])

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str, fmt).date()
            except (ValueError, TypeError):
                continue
        try:
            # Fallback for more complex but potentially valid date strings
            dt_obj = pd.to_datetime(date_str, errors="coerce")
            if pd.notna(dt_obj):
                return dt_obj.date()
        except Exception:
            pass  # Ignore if pandas also fails
        return None

    def _get_structured_entities_from_llm(self, query: str) -> dict:
        system_prompt = f"""
You are an expert entity extraction assistant. Your task is to analyze the user's query about sales data and extract key information.
You must return the information in a VALID JSON format.

The possible intents are:
- "brand_weekly_forecast_plot": If the user asks to forecast AND plot weekly sales/quantity for a specific brand.
- "sales_calculation": If the user is asking for historical sales figures (e.g., total sales) for a brand, region, or date period.
- "plot_data": If the user is asking to plot or visualize historical sales data for a SINGLE brand or a SINGLE region (or general data).
- "multi_brand_plot": If the user asks to plot or compare MULTIPLE brands on a single chart.
- "multi_region_plot": If the user asks to plot or compare MULTIPLE regions for a SINGLE brand on a single chart.
- "general_query": If the query does not fit the above or is a general question.

Valid brand names: {json.dumps(self.known_brands)}
Valid region names (branches) ARE CASE-SENSITIVE AND EXPECTED IN UPPERCASE: {json.dumps(self.known_regions)}
Possible plot metrics (for historical plots): "value" (for sales amount/revenue), "quantity" (for number of items).
Forecast plot metric (for brand_weekly_forecast_plot) is implicitly 'quantity' or 'sales' based on weekly aggregation.

Instructions:
1.  Determine the user's intent.
2.  For "plot_data", "brand_weekly_forecast_plot", "sales_calculation", and "multi_region_plot":
    Extract the SINGLE brand name. If a brand is mentioned that is not in the valid list, autocorrect it to the closest match. If no brand is mentioned or identifiable, use null.
3.  For "multi_brand_plot":
    Extract a LIST of brand names. Autocorrect each brand name to the closest match from the valid list. If no brands are mentioned or identifiable, use null for the list.
4.  For "plot_data", "sales_calculation", and "multi_brand_plot":
    Extract the SINGLE region name. If a region is mentioned that is not in the valid list, autocorrect it to the closest match. If no region is mentioned or identifiable, use null.
5.  For "multi_region_plot":
    Extract a LIST of region names. Autocorrect each region name to the closest match from the valid list. If no regions are mentioned or identifiable, use null for the list.
6.  Extract dates (for sales_calculation, plot_data, multi_brand_plot, multi_region_plot).
    - If a date range is mentioned, populate "start_date" and "end_date".
    - If a single historical date is mentioned, populate both "start_date" and "end_date" with that date.
    - Always parse and format dates as "YYYY-MM-DD".
    - If no date is mentioned or identifiable for a field, use null for that date field.
7.  If the intent is "plot_data", "multi_brand_plot", or "multi_region_plot", determine the metric to plot ("value" or "quantity"). Populate "plot_metric". If not specified, default to "value".

JSON Output Structure:
{{
  "intent": "brand_weekly_forecast_plot | sales_calculation | plot_data | multi_brand_plot | multi_region_plot | general_query",
  "brand": "SINGLE_CORRECTED_BRAND_NAME | null",
  "brands": ["CORRECTED_BRAND1", "CORRECTED_BRAND2"] | null,
  "region": "SINGLE_CORRECTED_REGION_NAME | null",
  "regions": ["CORRECTED_REGION1", "CORRECTED_REGION2"] | null,
  "target_date": null,
  "start_date": "YYYY-MM-DD | null",
  "end_date": "YYYY-MM-DD | null",
  "plot_metric": "value | quantity | null"
}}

Ensure the output is ONLY the JSON object.
User Query: "{query}"
JSON Output:
"""
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Using a capable model for complex extraction
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,  # Low temperature for deterministic output
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            # Clean potential markdown formatting if LLM wraps JSON in it
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            entities = json.loads(content)

            # Ensure target_date is always null as prophet_forecast is removed
            entities["target_date"] = None

            # Ensure correct fields are populated based on intent, and others are None
            intent = entities.get("intent")
            if intent == "multi_brand_plot":
                entities["brand"] = None  # Single brand not applicable
                entities["regions"] = None  # Multiple regions not applicable here
            elif intent == "multi_region_plot":
                entities["brands"] = None  # Multiple brands not applicable here
                entities["region"] = (
                    None  # Single region not applicable here (it's part of the list)
                )
            elif intent in [
                "plot_data",
                "brand_weekly_forecast_plot",
                "sales_calculation",
            ]:
                entities["brands"] = None  # Multiple brands not applicable
                entities["regions"] = None  # Multiple regions not applicable
            else:  # general_query or other unhandled intents
                entities["brands"] = None
                entities["regions"] = None
                entities["brand"] = None
                entities["region"] = None

            # Default plot_metric if applicable and missing
            if intent in [
                "plot_data",
                "multi_brand_plot",
                "multi_region_plot",
            ] and not entities.get("plot_metric"):
                entities["plot_metric"] = "value"

            return entities
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM: {e}\nLLM Output: {content}")
            # Return a default structure on error
            return {
                "intent": "general_query",
                "brand": None,
                "brands": None,
                "region": None,
                "regions": None,
                "target_date": None,
                "start_date": None,
                "end_date": None,
                "plot_metric": None,
            }
        except Exception as e:
            print(f"Error in _get_structured_entities_from_llm: {e}")
            return {
                "intent": "general_query",
                "brand": None,
                "brands": None,
                "region": None,
                "regions": None,
                "target_date": None,
                "start_date": None,
                "end_date": None,
                "plot_metric": None,
            }

    def _get_weekly_brand_sales_pd(
        self, brand_name: str, metric_col: str = "quantity"
    ) -> pd.DataFrame | str:
        if (
            "date" not in self.pl_df.columns
            or "brand" not in self.pl_df.columns
            or metric_col not in self.pl_df.columns
        ):
            return f"Error: Required columns ('date', 'brand', '{metric_col}') not found in the main data."

        brand_data_pl = self.pl_df.filter(pl.col("brand") == brand_name)
        if brand_data_pl.is_empty():
            return f"No data found for brand '{brand_name}'."

        brand_data_pd = brand_data_pl.select(["date", metric_col]).to_pandas()
        brand_data_pd["date"] = pd.to_datetime(brand_data_pd["date"])
        brand_data_pd = brand_data_pd.set_index("date")

        # Resample to weekly, summing the metric
        weekly_sales_pd = brand_data_pd[metric_col].resample("W").sum().reset_index()
        weekly_sales_pd.rename(
            columns={"date": "Time_Duration", metric_col: "Total_Sales"}, inplace=True
        )

        # Ensure Total_Sales is numeric, coercing errors and filling NaNs
        weekly_sales_pd["Total_Sales"] = pd.to_numeric(
            weekly_sales_pd["Total_Sales"], errors="coerce"
        ).fillna(0)

        if len(weekly_sales_pd) < 10:  # Minimum data for reliable forecast
            return f"Insufficient weekly data for brand '{brand_name}' to generate a reliable forecast (found {len(weekly_sales_pd)} weeks)."

        return weekly_sales_pd

    def _handle_outliers_pd(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series | None]:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (series < lower_bound) | (series > upper_bound)
        clean_series = series.copy()

        identified_outliers = None
        if outliers_mask.any():
            identified_outliers = series[outliers_mask]
            # Replace outliers with NaN, then interpolate
            clean_series[outliers_mask] = np.nan
            # Linear interpolation, then ffill/bfill for any remaining NaNs at ends
            clean_series = clean_series.interpolate(method="linear").ffill().bfill()
        return clean_series, identified_outliers

    def _forecast_brand_weekly_sales_plot(
        self, brand_name: str, forecast_steps: int = 4, metric_col: str = "quantity"
    ):
        weekly_sales_data_or_error = self._get_weekly_brand_sales_pd(
            brand_name, metric_col
        )
        if isinstance(weekly_sales_data_or_error, str):
            return weekly_sales_data_or_error

        weekly_sales_df = weekly_sales_data_or_error
        original_sales_series = weekly_sales_df.set_index("Time_Duration")[
            "Total_Sales"
        ]
        clean_sales_series, outliers = self._handle_outliers_pd(original_sales_series)

        if clean_sales_series.isna().any():  # Should be handled by _handle_outliers_pd
            clean_sales_series = clean_sales_series.ffill().bfill()

        min_required_data_for_seasonal_12 = 15
        if (
            clean_sales_series.empty
            or len(clean_sales_series) < min_required_data_for_seasonal_12
        ):
            return f"Not enough data for brand '{brand_name}' to fit Holt-Winters model with seasonal_periods=12 (need at least {min_required_data_for_seasonal_12} weeks, found {len(clean_sales_series)})."

        seasonal_types = ["add", "mul"]
        fixed_seasonal_periods = [12]  # As per new_model.py

        param_grid_seasonal = list(product(seasonal_types, fixed_seasonal_periods))

        best_model_fit = None
        best_mae = float("inf")
        best_params_combo = {}

        def _evaluate_smoothing_params(
            seasonal_type_eval, seasonal_period_eval, series_to_fit
        ):
            try:
                model_eval = ExponentialSmoothing(
                    series_to_fit,
                    trend="add",  # Assuming additive trend as common
                    seasonal=seasonal_type_eval,
                    seasonal_periods=seasonal_period_eval,
                    initialization_method="estimated",  # Common robust choice
                )

                smoothing_param_grid_ranges = (
                    slice(0.1, 0.9, 0.1),  # alpha
                    slice(0.1, 0.9, 0.1),  # beta (for trend)
                    slice(0.1, 0.9, 0.1),  # gamma (for seasonal)
                )

                def objective_aic(params_brute):
                    try:
                        return model_eval.fit(
                            smoothing_level=params_brute[0],
                            smoothing_trend=params_brute[1],
                            smoothing_seasonal=params_brute[2],
                            optimized=True,  # As in new_model.py
                            remove_bias=True,  # As in new_model.py
                        ).aic
                    except Exception:
                        return float("inf")  # Penalize failures

                optimal_smoothing_params = brute(
                    objective_aic,
                    smoothing_param_grid_ranges,
                    finish=None,  # No local optimization after brute
                    disp=False,
                )

                fitted_model_eval = model_eval.fit(
                    smoothing_level=optimal_smoothing_params[0],
                    smoothing_trend=optimal_smoothing_params[1],
                    smoothing_seasonal=optimal_smoothing_params[2],
                    optimized=True,
                    remove_bias=True,
                )

                # Align fitted values with original series index for MAE calculation
                aligned_fitted_values = fitted_model_eval.fittedvalues.reindex(
                    series_to_fit.index
                )
                mae_eval = mean_absolute_error(
                    series_to_fit,
                    aligned_fitted_values.fillna(method="ffill").fillna(method="bfill"),
                )

                return mae_eval, fitted_model_eval, optimal_smoothing_params
            except Exception:
                return float("inf"), None, None

        for seasonal_type_outer, seasonal_period_outer in param_grid_seasonal:
            current_mae, current_fit, current_smoothing_params = (
                _evaluate_smoothing_params(
                    seasonal_type_outer, seasonal_period_outer, clean_sales_series
                )
            )

            if current_fit is not None and current_mae < best_mae:
                best_mae = current_mae
                best_model_fit = current_fit
                best_params_combo = {
                    "seasonal_type": seasonal_type_outer,
                    "seasonal_period": seasonal_period_outer,
                    "smoothing_params": current_smoothing_params,
                    "trend": "add",  # Explicitly stating trend used
                }

        if best_model_fit is None:
            return (
                f"Could not fit any Holt-Winters forecasting model for {brand_name} "
                f"using the specified parameters (seasonal types: {seasonal_types}, periods: {fixed_seasonal_periods}). "
                "The data might be too short, too erratic, or incompatible with these settings."
            )

        fit_model = best_model_fit
        forecast_values = fit_model.forecast(steps=forecast_steps)
        last_date = clean_sales_series.index.max()
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1), periods=forecast_steps, freq="W"
        )
        forecast_series = pd.Series(forecast_values, index=forecast_index)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=original_sales_series.index,
                y=original_sales_series,
                mode="lines+markers",
                name="Real Weekly Sales (Original)",
                line=dict(color="blue"),
            )
        )
        if outliers is not None and not outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=outliers.index,
                    y=outliers,
                    mode="markers",
                    name="Identified Outliers",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=clean_sales_series.index,  # Plot fit against cleaned data
                y=fit_model.fittedvalues,
                mode="lines",
                name="Model Fit on Historical (Cleaned)",
                line=dict(color="purple", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                mode="lines+markers",
                name="Forecasted Weekly Sales",
                line=dict(color="green", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Weekly Sales Forecast for {brand_name} ({metric_col.capitalize()})",
            xaxis_title="Date (Week Ending)",
            yaxis_title=metric_col.capitalize(),
            legend_title="Legend",
        )
        return {"type": "plot", "figure": fig}

    def _generate_timeseries_plot(
        self,
        brand: str | None,
        region: str | None,
        start_date: date | None,
        end_date: date | None,
        plot_metric: str | None,
    ):
        if "date" not in self.pl_df.columns:
            return "Plotting error: 'date' column is missing from the data."
        if not plot_metric or plot_metric not in ["value", "quantity"]:
            return f"Plotting error: Invalid or missing metric to plot. Choose 'value' or 'quantity'."
        if plot_metric not in self.pl_df.columns:
            return (
                f"Plotting error: Metric column '{plot_metric}' not found in the data."
            )

        filters = []
        title_parts = [f"{plot_metric.capitalize()} Trend"]

        temp_df = self.pl_df

        if brand:
            if "brand" not in temp_df.columns:
                return "Plotting error: 'brand' column missing."
            filters.append(pl.col("brand") == brand)
            title_parts.append(f"for Brand '{brand}'")
        if region:
            if "branch" not in temp_df.columns:
                return "Plotting error: 'branch' column missing."
            filters.append(pl.col("branch") == region)
            title_parts.append(f"in Region '{region}'")

        if start_date and end_date:
            filters.append(
                pl.col("date").is_between(start_date, end_date, closed="both")
            )
            title_parts.append(
                f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
        elif start_date:
            filters.append(pl.col("date") >= start_date)
            title_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
        elif end_date:
            filters.append(pl.col("date") <= end_date)
            title_parts.append(f"up to {end_date.strftime('%Y-%m-%d')}")

        if filters:
            final_filter = filters[0]
            for f_idx in range(1, len(filters)):
                final_filter = final_filter & filters[f_idx]
            temp_df = temp_df.filter(final_filter)

        if temp_df.is_empty():
            return f"No data available to plot for the specified criteria: {', '.join(title_parts[1:]) if len(title_parts) > 1 else 'all data'}."

        try:
            plot_df_pl = (
                temp_df.group_by("date")
                .agg(pl.col(plot_metric).sum().alias(plot_metric))
                .sort("date")
            )

            if plot_df_pl.is_empty():
                return f"No aggregated data to plot for {', '.join(title_parts[1:]) if len(title_parts) > 1 else 'all data'}."

            plot_df_pd = plot_df_pl.to_pandas()

            if plot_df_pd.empty:
                return f"No data to plot after conversion for {', '.join(title_parts[1:]) if len(title_parts) > 1 else 'all data'}."

            fig = px.line(
                plot_df_pd, x="date", y=plot_metric, title=" ".join(title_parts)
            )
            fig.update_layout(xaxis_title="Date", yaxis_title=plot_metric.capitalize())
            return {"type": "plot", "figure": fig}
        except Exception as e:
            return f"Error generating plot: {str(e)}"

    def _generate_multi_brand_plot(
        self,
        brands: list[str] | None,
        region: str | None,
        start_date: date | None,
        end_date: date | None,
        plot_metric: str | None,
    ):
        if not brands:
            return "Plotting error: No brands specified for multi-brand comparison."
        if "date" not in self.pl_df.columns:
            return "Plotting error: 'date' column is missing from the data."
        if not plot_metric or plot_metric not in ["value", "quantity"]:
            return f"Plotting error: Invalid or missing metric. Choose 'value' or 'quantity'."
        if plot_metric not in self.pl_df.columns:
            return f"Plotting error: Metric column '{plot_metric}' not found."
        if region and "branch" not in self.pl_df.columns:
            return "Plotting error: 'branch' column missing for region filtering."
        if "brand" not in self.pl_df.columns:
            return "Plotting error: 'brand' column missing for brand filtering."

        fig = go.Figure()
        title_parts = [f"{plot_metric.capitalize()} Comparison"]
        data_found_for_at_least_one_brand = False

        base_filters = []
        if region:
            base_filters.append(pl.col("branch") == region)
            title_parts.append(f"in Region '{region}'")

        date_filter_applied = False
        if start_date and end_date:
            base_filters.append(
                pl.col("date").is_between(start_date, end_date, closed="both")
            )
            title_parts.append(
                f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            date_filter_applied = True
        elif start_date:
            base_filters.append(pl.col("date") >= start_date)
            title_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
            date_filter_applied = True
        elif end_date:
            base_filters.append(pl.col("date") <= end_date)
            title_parts.append(f"up to {end_date.strftime('%Y-%m-%d')}")
            date_filter_applied = True

        brand_list_str = ", ".join(brands)
        # Insert brand list after "Comparison" but before other filters like region/date
        title_parts.insert(1, f"for Brands: {brand_list_str}")

        for current_brand in brands:
            current_brand_filter = pl.col("brand") == current_brand
            all_filters_for_brand = [
                current_brand_filter
            ] + base_filters  # Brand filter first

            final_filter_expr = all_filters_for_brand[0]
            for i in range(1, len(all_filters_for_brand)):
                final_filter_expr = final_filter_expr & all_filters_for_brand[i]

            temp_df = self.pl_df.filter(final_filter_expr)

            if not temp_df.is_empty():
                plot_df_pl = (
                    temp_df.group_by("date")
                    .agg(pl.col(plot_metric).sum().alias(plot_metric))
                    .sort("date")
                )
                if not plot_df_pl.is_empty():
                    plot_df_pd = plot_df_pl.to_pandas()
                    if not plot_df_pd.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df_pd["date"],
                                y=plot_df_pd[plot_metric],
                                mode="lines+markers",
                                name=current_brand,
                            )
                        )
                        data_found_for_at_least_one_brand = True

        if not data_found_for_at_least_one_brand:
            # Construct a more informative message
            criteria_desc_parts = []
            if region:
                criteria_desc_parts.append(f"region '{region}'")
            if date_filter_applied:
                if start_date and end_date:
                    criteria_desc_parts.append(
                        f"period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                elif start_date:
                    criteria_desc_parts.append(
                        f"period from {start_date.strftime('%Y-%m-%d')}"
                    )
                elif end_date:
                    criteria_desc_parts.append(
                        f"period up to {end_date.strftime('%Y-%m-%d')}"
                    )

            criteria_str = " and ".join(criteria_desc_parts)
            if criteria_str:
                return f"No data available to plot for brands {brand_list_str} with criteria: {criteria_str}."
            else:
                return f"No data available to plot for brands {brand_list_str}."

        fig.update_layout(
            title=" ".join(title_parts),
            xaxis_title="Date",
            yaxis_title=plot_metric.capitalize(),
            legend_title="Brands",
        )
        return {"type": "plot", "figure": fig}

    def _generate_multi_region_plot(
        self,
        brand: str | None,
        regions: list[str] | None,
        start_date: date | None,
        end_date: date | None,
        plot_metric: str | None,
    ):
        if not regions:
            return "Plotting error: No regions specified for multi-region comparison."
        if not brand:  # Brand is mandatory for this plot type
            return "Plotting error: No brand specified for multi-region comparison."
        if "date" not in self.pl_df.columns:
            return "Plotting error: 'date' column is missing from the data."
        if not plot_metric or plot_metric not in ["value", "quantity"]:
            return f"Plotting error: Invalid or missing metric. Choose 'value' or 'quantity'."
        if plot_metric not in self.pl_df.columns:
            return f"Plotting error: Metric column '{plot_metric}' not found."
        if "branch" not in self.pl_df.columns:  # branch is used for region
            return "Plotting error: 'branch' column missing for region filtering."
        if "brand" not in self.pl_df.columns:
            return "Plotting error: 'brand' column missing for brand filtering."

        fig = go.Figure()
        title_parts = [f"{plot_metric.capitalize()} Comparison for Brand '{brand}'"]
        data_found_for_at_least_one_region = False

        # Base filter is always the brand
        base_filters = [pl.col("brand") == brand]

        date_filter_applied = False
        if start_date and end_date:
            base_filters.append(
                pl.col("date").is_between(start_date, end_date, closed="both")
            )
            title_parts.append(
                f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            date_filter_applied = True
        elif start_date:
            base_filters.append(pl.col("date") >= start_date)
            title_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
            date_filter_applied = True
        elif end_date:
            base_filters.append(pl.col("date") <= end_date)
            title_parts.append(f"up to {end_date.strftime('%Y-%m-%d')}")
            date_filter_applied = True

        region_list_str = ", ".join(regions)
        title_parts.insert(1, f"across Regions: {region_list_str}")

        for current_region in regions:
            current_region_filter = pl.col("branch") == current_region
            # Combine region filter with base filters (which includes brand and optionally date)
            all_filters_for_region = [current_region_filter] + base_filters

            final_filter_expr = all_filters_for_region[0]
            for i in range(1, len(all_filters_for_region)):
                final_filter_expr = final_filter_expr & all_filters_for_region[i]

            temp_df = self.pl_df.filter(final_filter_expr)

            if not temp_df.is_empty():
                plot_df_pl = (
                    temp_df.group_by("date")
                    .agg(pl.col(plot_metric).sum().alias(plot_metric))
                    .sort("date")
                )
                if not plot_df_pl.is_empty():
                    plot_df_pd = plot_df_pl.to_pandas()
                    if not plot_df_pd.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df_pd["date"],
                                y=plot_df_pd[plot_metric],
                                mode="lines+markers",
                                name=current_region,
                            )
                        )
                        data_found_for_at_least_one_region = True

        if not data_found_for_at_least_one_region:
            criteria_desc_parts = []
            if date_filter_applied:
                if start_date and end_date:
                    criteria_desc_parts.append(
                        f"period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                elif start_date:
                    criteria_desc_parts.append(
                        f"period from {start_date.strftime('%Y-%m-%d')}"
                    )
                elif end_date:
                    criteria_desc_parts.append(
                        f"period up to {end_date.strftime('%Y-%m-%d')}"
                    )

            criteria_str = " and ".join(criteria_desc_parts)
            if criteria_str:
                return f"No data available to plot for Brand '{brand}' across regions {region_list_str} with criteria: {criteria_str}."
            else:
                return f"No data available to plot for Brand '{brand}' across regions {region_list_str}."

        fig.update_layout(
            title=" ".join(title_parts),
            xaxis_title="Date",
            yaxis_title=plot_metric.capitalize(),
            legend_title="Regions",
        )
        return {"type": "plot", "figure": fig}

    def get_sales_for_criteria(
        self,
        brand: str | None,
        region: str | None,
        start_date: date | None,
        end_date: date | None,
    ) -> str:
        if "value" not in self.pl_df.columns:
            return "The 'value' column (for sales) is missing from the data."
        if "date" not in self.pl_df.columns and (start_date or end_date):
            return "The 'date' column is missing, cannot filter by period."
        if "brand" not in self.pl_df.columns and brand:
            return "The 'brand' column is missing, cannot filter by brand."
        if (
            "branch" not in self.pl_df.columns and region
        ):  # 'branch' is the column name for region
            return "The 'branch' (region) column is missing, cannot filter by region."

        filters = []
        description_parts = []

        if brand:
            filters.append(pl.col("brand") == brand)
            description_parts.append(f"for brand '{brand}'")
        if region:
            filters.append(pl.col("branch") == region)  # Use 'branch' column
            description_parts.append(f"in region '{region}'")

        if start_date and end_date:
            if (
                "date" not in self.pl_df.columns
            ):  # Should be caught earlier but good for robustness
                return "Cannot filter by date period as 'date' column is missing."
            filters.append(
                pl.col("date").is_between(start_date, end_date, closed="both")
            )
            if start_date == end_date:
                description_parts.append(f"on {start_date.strftime('%Y-%m-%d')}")
            else:
                description_parts.append(
                    f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                )
        elif start_date:
            filters.append(pl.col("date") >= start_date)
            description_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
        elif end_date:
            filters.append(pl.col("date") <= end_date)
            description_parts.append(f"up to {end_date.strftime('%Y-%m-%d')}")

        if not filters:  # If no criteria are provided at all
            return "Please specify at least one criterion (brand, region, or date period) for calculating sales."

        final_filter = filters[0]
        for f_idx in range(1, len(filters)):
            final_filter = final_filter & filters[f_idx]

        try:
            result_df = self.pl_df.filter(final_filter)
            if result_df.is_empty():
                return f"No sales data found {', '.join(description_parts) if description_parts else 'for the specified criteria'}."

            total_sales = result_df.select(
                pl.col("value").sum()
            ).item()  # Assuming 'value' is the sales metric
            return f"Total sales {', '.join(description_parts) if description_parts else 'for the specified criteria'} were ${total_sales:,.2f}."
        except Exception as e:
            return f"Error calculating sales: {str(e)}"

    def generate_response(self, query: str, history: list = None):
        extracted_data = self._get_structured_entities_from_llm(query)

        parsed_intent = extracted_data.get("intent")
        parsed_brand = extracted_data.get("brand")
        parsed_brands = extracted_data.get("brands")

        raw_region_from_llm = extracted_data.get("region")
        parsed_region = normalize_region_name(raw_region_from_llm)

        raw_regions_from_llm = extracted_data.get("regions")
        parsed_regions = normalize_region_list(raw_regions_from_llm)

        llm_start_date_str = extracted_data.get("start_date")
        llm_end_date_str = extracted_data.get("end_date")
        parsed_plot_metric = extracted_data.get(
            "plot_metric"
        )  # LLM should default this for plot intents

        parsed_start_date = self._parse_date(llm_start_date_str)
        parsed_end_date = self._parse_date(llm_end_date_str)

        if parsed_intent == "brand_weekly_forecast_plot":
            if parsed_brand:
                forecast_metric = "quantity"  # Default for this specific forecast type
                return self._forecast_brand_weekly_sales_plot(
                    brand_name=parsed_brand, metric_col=forecast_metric
                )
            else:
                return (
                    "Please specify a brand to generate a weekly sales forecast plot."
                )

        elif parsed_intent == "sales_calculation":
            if parsed_brand or parsed_region or (parsed_start_date and parsed_end_date):
                return self.get_sales_for_criteria(
                    parsed_brand, parsed_region, parsed_start_date, parsed_end_date
                )
            else:
                # Not enough criteria for a specific calculation, treat as general query
                parsed_intent = "general_query"

        elif parsed_intent == "plot_data":  # Single brand/region plot
            # plot_metric should be defaulted by LLM or here
            if not parsed_plot_metric:
                parsed_plot_metric = "value"
            return self._generate_timeseries_plot(
                parsed_brand,
                parsed_region,
                parsed_start_date,
                parsed_end_date,
                parsed_plot_metric,
            )

        elif parsed_intent == "multi_brand_plot":
            if not parsed_plot_metric:
                parsed_plot_metric = "value"  # Default if somehow missed
            if (
                parsed_brands
                and isinstance(parsed_brands, list)
                and len(parsed_brands) > 0
            ):
                return self._generate_multi_brand_plot(
                    brands=parsed_brands,
                    region=parsed_region,  # Single region is optional
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    plot_metric=parsed_plot_metric,
                )
            else:
                return "Please specify at least one brand (preferably two or more for comparison) for the multi-brand plot."

        elif parsed_intent == "multi_region_plot":
            if not parsed_plot_metric:
                parsed_plot_metric = "value"  # Default if somehow missed
            if (
                parsed_brand
                and parsed_regions
                and isinstance(parsed_regions, list)
                and len(parsed_regions) > 0
            ):
                return self._generate_multi_region_plot(
                    brand=parsed_brand,  # Single brand is required
                    regions=parsed_regions,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    plot_metric=parsed_plot_metric,
                )
            else:
                return "Please specify a brand and at least one region (preferably two or more for comparison) for the multi-region plot."

        # Fallback to General LLM flow if intent is "general_query" or becomes one
        history_text = ""
        if history:
            for user_msg, bot_msg in history:
                history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        general_prompt = f"""You are a sales data assistant. Answer the user's question based on the provided comprehensive sales data summary and the conversation history.
If the question asks for specific calculations like total sales for a certain brand, region, or period, and the information is not directly in the summary, state that you cannot perform that specific calculation from the summary but can answer based on the general trends provided.
If asked for a weekly brand forecast plot, a multi-brand plot, or a multi-region plot, and you have provided one, state it clearly. Otherwise, explain that it might not be available or an error occurred.

Comprehensive Sales Data Summary:
{self.full_context_text}

Conversation history:
{history_text}User: {query}
Assistant:
"""
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful sales data assistant. Use the provided comprehensive summary to answer questions."
                        " If the summary doesn't contain the answer for a specific calculation, say so. "
                        "You can perform specific sales calculations, brand-specific weekly forecast plots, multi-brand comparison plots, and multi-region comparison plots if requested and possible."
                        f"The branch (region) can only be of the types: {json.dumps(self.known_regions)}, "
                        "so autocorrect anything different to the closest match out of these. These words may not be always placed after branch. "
                        f"Similarly, the brand can only be of the types: {json.dumps(self.known_brands)}, "
                        "so autocorrect anything different to the closest match out of these,RETURNED IN UPPERCASE. These words may not be always placed after brand. "
                        "When dates are mentioned in natural language (e.g., '25th May 2025', '25 May 2025', 'May 25 2025'), "
                        "always parse and refer to them in the YYYY-MM-DD format (e.g., '2025-05-25').",
                    },
                    {"role": "user", "content": general_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return (
                f"Sorry, I encountered an error while generating the response: {str(e)}"
            )

    def answer_query(self, query: str, history: list = None):
        # This method can be simplified if it's just a pass-through
        # or can include pre/post processing if needed in the future.
        response = self.generate_response(query, history=history)
        return response

    # These helper methods for entity/date extraction are not currently used
    # by the LLM-based _get_structured_entities_from_llm, but are kept for potential future use
    # or alternative extraction strategies.
    def _extract_entity_from_query(
        self, query: str, entity_keywords: list[str], known_entities: list[str] = None
    ) -> str | None:
        # Basic keyword-based extraction (example, not robust)
        # This is superseded by the LLM's extraction capabilities.
        query_lower = query.lower()
        for keyword in entity_keywords:
            if keyword in query_lower:
                # This would need more sophisticated logic to find the actual entity value
                # and match against known_entities.
                pass  # Placeholder
        return None

    def _extract_date_range_from_query(
        self, query: str
    ) -> tuple[date | None, date | None]:
        # Basic date extraction (example, not robust)
        # This is superseded by the LLM's extraction and _parse_date.
        # Would require libraries like dateparser or complex regex.
        return None, None


if __name__ == "__main__":
    # Example usage (optional, for direct testing of the class)
    # Create a dummy parquet file for testing if needed
    if not os.path.exists("dummy_sales_rt.parquet"):
        print("Creating dummy sales data for testing...")
        num_rows = 1000
        start_d = date(2023, 1, 1)
        brands_list = ["INDOMIE PULL", "POWER OIL ", "DANO MILK", "COLGATE MAX"]
        regions_list = ["NORTH 1", "SOUTH WEST", "EASTERN", "CENTRAL"]

        data = {
            "date": [start_d + timedelta(days=i % 365) for i in range(num_rows)],
            "brand": [brands_list[i % len(brands_list)] for i in range(num_rows)],
            "branch": [regions_list[i % len(regions_list)] for i in range(num_rows)],
            "value": [np.random.uniform(100, 10000) for _ in range(num_rows)],
            "quantity": [np.random.randint(1, 50) for _ in range(num_rows)],
            "order_id": [f"ORD{1000 + i}" for i in range(num_rows)],
            "sku_code": [f"SKU{100 + i % 50}" for i in range(num_rows)],
            "sku_name": [f"Product {i % 50 + 1}" for i in range(num_rows)],
        }
        dummy_df_pd = pd.DataFrame(data)
        dummy_df_pl = pl.from_pandas(dummy_df_pd)
        dummy_df_pl.write_parquet("dummy_sales_rt.parquet")
        print("Dummy sales data created: dummy_sales_rt.parquet")

    data_file_path = "dummy_sales_rt.parquet"  # Or your actual data file
    if (
        not os.path.exists(data_file_path)
        and data_file_path == "dummy_sales_rt.parquet"
    ):
        print(
            f"Error: Dummy data file '{data_file_path}' not found. Run script once to create it."
        )
    elif not os.path.exists(data_file_path):
        print(
            f"Error: Data file '{data_file_path}' not found. Please provide a valid path."
        )
    else:
        chatbot = SalesDataChatbot(data_path=data_file_path)
        print("\nSalesDataChatbot initialized.")
        print(f"Known Brands: {chatbot.known_brands}")
        print(f"Known Regions: {chatbot.known_regions}")
        print("-" * 30)

        test_queries = [
            "What were the total sales for INDOMIE PULL in NORTH 1 last year?",  # Needs date parsing logic in LLM
            "Plot sales for DANO MILK",
            "Forecast weekly quantity for POWER OIL",
            "Show me a plot comparing INDOMIE PULL and DANO MILK for value in SOUTH WEST",
            "Compare sales for POWER OIL in NORTH 1 and EASTERN from 2023-03-01 to 2023-05-30",
            "What is the trend for COLGATE MAX quantity?",
            "Any cross-selling opportunities?",
            "Summarize performance for INDOMIE PULL",
        ]

        for q in test_queries:
            print(f"\nUser Query: {q}")
            response = chatbot.answer_query(q)
            if isinstance(response, dict) and response.get("type") == "plot":
                print(
                    "Chatbot Response: (Plotly figure generated, not displayed in console)"
                )
                # response["figure"].show() # Uncomment to display plots if running locally
            else:
                print(f"Chatbot Response: {response}")
            print("-" * 30)
