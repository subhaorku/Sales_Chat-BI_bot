import polars as pl


class PerformanceAnalyzer:
    def __init__(self, df: pl.DataFrame):
        """
        Initialize with transaction data.

        Args:
            df (pl.DataFrame): Polars DataFrame with columns like
                `brand`, `sku_name`, `customer_name`, `branch`, `date`, `quantity`, `value`.
        """
        self.df = df
        self.summaries = []
        self._generate_summaries()

    def _generate_summaries(self):
        """Precompute text summaries of top performers."""
        if self.df.is_empty():
            self.summaries.append(
                "No performance data available to generate summaries."
            )
            return

        # 1. Top brands globally
        if "brand" in self.df.columns:
            # By value
            if "value" in self.df.columns:
                top_brands_value = self._get_top_performers(
                    group_by="brand", based_on="value", n=10
                )
                if top_brands_value:
                    self.summaries.append(
                        "Top brands by revenue:\n"
                        + "\n".join(
                            f"- {row['brand']}: ${row['value']:,.2f}"
                            for row in top_brands_value
                        )
                    )
                else:
                    self.summaries.append("Could not determine top brands by revenue.")
            else:
                self.summaries.append(
                    "Missing 'value' column for top brands by revenue summary."
                )

            # By quantity
            if "quantity" in self.df.columns:
                top_brands_quantity = self._get_top_performers(
                    group_by="brand", based_on="quantity", n=10
                )
                if top_brands_quantity:
                    self.summaries.append(
                        "Top brands by quantity sold:\n"
                        + "\n".join(
                            f"- {row['brand']}: {int(row['quantity']):,} units"
                            for row in top_brands_quantity
                        )
                    )
                else:
                    self.summaries.append("Could not determine top brands by quantity.")
            else:
                self.summaries.append(
                    "Missing 'quantity' column for top brands by quantity summary."
                )
        else:
            self.summaries.append("Missing 'brand' column for top brands summary.")

        # 2. Top customers globally (by quantity) - Preserved
        if "customer_name" in self.df.columns and "quantity" in self.df.columns:
            top_customers = self._get_top_performers(
                group_by="customer_name", based_on="quantity", n=10
            )
            if top_customers:
                self.summaries.append(
                    "Top customers by volume:\n"
                    + "\n".join(
                        f"- {row['customer_name']}: {int(row['quantity']):,} units"
                        for row in top_customers
                    )
                )
            else:
                self.summaries.append("Could not determine top customers by volume.")
        else:
            self.summaries.append(
                "Missing 'customer_name' or 'quantity' column for top customers summary."
            )

        # 3. Top regions (by value) - Preserved
        if "branch" in self.df.columns and "value" in self.df.columns:
            top_regions = self._get_top_performers(
                group_by="branch", based_on="value", n=15
            )
            if top_regions:
                self.summaries.append(
                    "Top regions by revenue:\n"
                    + "\n".join(
                        f"- {row['branch']}: ${row['value']:,.2f}"
                        for row in top_regions
                    )
                )
            else:
                self.summaries.append("Could not determine top regions by revenue.")
        else:
            self.summaries.append(
                "Missing 'branch' or 'value' column for top regions summary."
            )

        # 4. Top SKUs globally (New)
        if "sku_name" in self.df.columns:
            # By value
            if "value" in self.df.columns:
                top_skus_value = self._get_top_performers(
                    group_by="sku_name", based_on="value", n=10
                )
                if top_skus_value:
                    self.summaries.append(
                        "Top SKUs by revenue:\n"
                        + "\n".join(
                            f"- {row['sku_name']}: ${row['value']:,.2f}"
                            for row in top_skus_value
                        )
                    )
                else:
                    self.summaries.append("Could not determine top SKUs by revenue.")
            else:
                self.summaries.append(
                    "Missing 'value' column for top SKUs by revenue summary."
                )

            # By quantity
            if "quantity" in self.df.columns:
                top_skus_quantity = self._get_top_performers(
                    group_by="sku_name", based_on="quantity", n=10
                )
                if top_skus_quantity:
                    self.summaries.append(
                        "Top SKUs by quantity sold:\n"
                        + "\n".join(
                            f"- {row['sku_name']}: {int(row['quantity']):,} units"
                            for row in top_skus_quantity
                        )
                    )
                else:
                    self.summaries.append("Could not determine top SKUs by quantity.")
            else:
                self.summaries.append(
                    "Missing 'quantity' column for top SKUs by quantity summary."
                )
        else:
            self.summaries.append("Missing 'sku_name' column for top SKUs summary.")

    def _get_top_performers(
        self, group_by: str, based_on: str = "value", n: int = 10, date_range=None
    ):
        """Helper: Compute top performers for a category."""
        if group_by not in self.df.columns or based_on not in self.df.columns:
            return []

        query_df = self.df
        if date_range and "date" in self.df.columns:
            try:
                query_df = query_df.filter(
                    pl.col("date").is_between(
                        date_range[0], date_range[1], closed="both"
                    )
                )
            except Exception:  # Handle potential errors if date_range is not valid or date column issues
                pass  # Or log a warning

        if query_df.is_empty():
            return []

        return (
            query_df.group_by([group_by])
            .agg(
                pl.col(based_on).sum().fill_null(0)
            )  # fill_null for sum if all values in a group are null
            .sort(based_on, descending=True)
            .limit(n)
            .to_dicts()
        )
