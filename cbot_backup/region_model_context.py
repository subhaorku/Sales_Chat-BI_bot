import polars as pl


class CustomerAnalyzer:
    def __init__(self, df: pl.DataFrame):
        """
        Initialize with transaction data.

        Args:
            df (pl.DataFrame): Polars DataFrame with columns like:
                `customer_code`, `customer_name`, `branch`, `date`,
                `quantity`, `value`.
        """
        if "branch" in df.columns:
            self.df = df.filter(
                pl.col("branch").is_not_null() & (pl.col("branch") != "")
            )
        else:
            self.df = df
        self.summaries = []
        self._generate_summaries()

    def _generate_summaries(self):
        """Precompute text summaries of key insights."""
        if self.df.is_empty():
            self.summaries.append(
                "No customer data available to generate regional summaries."
            )
            return

        # 1. Top customers globally by revenue
        if "customer_name" in self.df.columns and "value" in self.df.columns:
            top_global = self._get_top_customers(based_on="value", region="All Regions")
            if top_global:
                self.summaries.append(
                    "Global top customers by revenue:\n"
                    + "\n".join(
                        f"{row['customer_name']} (${row['value']:,.2f})"
                        for row in top_global
                    )
                )
            else:
                self.summaries.append(
                    "Could not determine global top customers by revenue."
                )
        else:
            self.summaries.append(
                "Missing 'customer_name' or 'value' for global top customers summary."
            )

        # 2. Regional top customers (example for some regions)
        if (
            "branch" in self.df.columns
            and "customer_name" in self.df.columns
            and "value" in self.df.columns
        ):
            unique_regions = self.df["branch"].unique().to_list()
            regions_to_summarize = unique_regions[:3]  # Limit to 3 regions for demo

            if not regions_to_summarize:
                self.summaries.append("No regions found to summarize top customers.")

            for region in regions_to_summarize:
                if region is None:
                    continue  # Skip if region is None
                top_region = self._get_top_customers(based_on="value", region=region)
                if top_region:
                    self.summaries.append(
                        f"Top customers in {region} by revenue:\n"
                        + "\n".join(
                            f"{row['customer_name']} (${row['value']:,.2f})"
                            for row in top_region
                        )
                    )
                else:
                    self.summaries.append(
                        f"Could not determine top customers for region: {region}."
                    )
        else:
            self.summaries.append(
                "Missing 'branch', 'customer_name', or 'value' for regional top customers summary."
            )

    def _get_top_customers(
        self, based_on: str = "value", region: str = "All Regions", n: int = 10
    ):
        """Helper: Compute top customers for a region/time period."""
        if not all(
            col in self.df.columns
            for col in ["customer_code", "customer_name", based_on, "branch"]
        ):
            return []

        query_df = self.df
        if region != "All Regions":
            query_df = query_df.filter(pl.col("branch") == region)

        if query_df.is_empty():
            return []

        return (
            query_df.group_by(
                ["customer_code"]
            )  # Assuming customer_code is unique identifier
            .agg(
                pl.col("customer_name").first(),
                pl.col(based_on).sum().fill_null(0),  # fill_null for sum
                pl.col("branch").first(),
            )
            .sort(based_on, descending=True)
            .limit(n)
            .to_dicts()
        )
