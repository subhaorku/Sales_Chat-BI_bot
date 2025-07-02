import polars as pl
from efficient_apriori import apriori


class CrossSellingAnalyzer:
    def __init__(
        self,
        pl_df: pl.DataFrame,
        order_id_col: str = "order_id",
        item_id_col: str = "sku_code",
        item_name_col: str = "sku_name",
        min_support=0.005,
        min_confidence=0.5,
        max_rule_length=5,
        top_n_rules_output=75,
    ):
        """
        Initialize with transaction data and Apriori parameters.
        Args:
            pl_df (pl.DataFrame): Polars DataFrame.
            order_id_col (str): Name of the column representing order/transaction IDs.
            item_id_col (str): Name of the column representing item/product IDs.
            item_name_col (str): Name of the column representing item/product names (optional).
            min_support (float): Minimum support for Apriori.
            min_confidence (float): Minimum confidence for Apriori.
            max_rule_length (int): Maximum number of items in a rule's itemset for Apriori.
            top_n_rules_output (int): Number of top rules to include in rules_text.
        """
        self.pl_df = pl_df
        self.order_id_col = order_id_col
        self.item_id_col = item_id_col
        self.item_name_col = item_name_col
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_rule_length = max_rule_length
        self.top_n_rules_output = top_n_rules_output

        self.product_name_map = self._generate_product_name_map()
        self.apriori_rules = self._run_apriori()
        self.rules_text = self._format_apriori_rules()
        self.summaries = self._generate_summaries()

    def _generate_product_name_map(self):
        product_map = {}
        if (
            self.item_id_col in self.pl_df.columns
            and self.item_name_col in self.pl_df.columns
        ):
            try:
                name_df = (
                    self.pl_df.select([self.item_id_col, self.item_name_col])
                    .drop_nulls(subset=[self.item_id_col])
                    .unique(subset=[self.item_id_col], keep="first")
                )
                for row in name_df.to_dicts():
                    item_id = row[self.item_id_col]
                    item_name = row[self.item_name_col]
                    if (
                        item_id is not None
                    ):  # Ensure item_id is not None before using as key
                        product_map[str(item_id)] = item_name
            except Exception as e:
                print(f"Error generating product name map: {e}")
        return product_map

    def _get_product_display_name(self, item_id):
        # item_id from apriori rules should be string if cast before apriori
        return self.product_name_map.get(str(item_id), str(item_id))

    def _run_apriori(self):
        rules_list = []
        required_cols = [self.order_id_col, self.item_id_col]
        if not all(col in self.pl_df.columns for col in required_cols):
            print(
                f"CrossSellingAnalyzer: Missing required columns for Apriori. Need '{self.order_id_col}' and '{self.item_id_col}'. Found columns: {self.pl_df.columns}"
            )
            return rules_list

        if self.pl_df.is_empty():
            print("CrossSellingAnalyzer: Input DataFrame is empty.")
            return rules_list

        # Ensure item_ids are strings for consistency in apriori and mapping
        temp_df = self.pl_df.select([self.order_id_col, self.item_id_col]).drop_nulls(
            subset=[self.order_id_col, self.item_id_col]
        )

        if temp_df[self.item_id_col].dtype != pl.Utf8:
            temp_df = temp_df.with_columns(
                pl.col(self.item_id_col).cast(pl.Utf8).alias(self.item_id_col)
            )

        if temp_df.is_empty():
            print(
                "CrossSellingAnalyzer: DataFrame is empty after selecting and dropping nulls from key columns."
            )
            return rules_list

        transactions_df = (
            temp_df.group_by(self.order_id_col)
            .agg(pl.col(self.item_id_col).unique())
            .filter(pl.col(self.item_id_col).list.len() > 1)
        )

        if transactions_df.is_empty():
            print(
                "CrossSellingAnalyzer: No transactions with more than one unique item found after filtering."
            )
            return rules_list

        transactions = [
            tuple(t_list) for t_list in transactions_df[self.item_id_col].to_list()
        ]

        if not transactions:
            print(
                "CrossSellingAnalyzer: Transaction list is empty before running Apriori."
            )
            return rules_list

        print(
            f"CrossSellingAnalyzer: Running Apriori on {len(transactions)} transactions. Min_support={self.min_support}, Min_confidence={self.min_confidence}, Max_length={self.max_rule_length}"
        )

        try:
            _, rules = apriori(
                transactions,
                min_support=self.min_support,
                min_confidence=self.min_confidence,
                max_length=self.max_rule_length,
            )
            rules_list = sorted(
                list(rules), key=lambda r: (r.confidence, r.support), reverse=True
            )
            print(f"CrossSellingAnalyzer: Apriori found {len(rules_list)} rules.")
        except Exception as e:
            print(f"CrossSellingAnalyzer: Error during Apriori execution: {e}")
            return []

        return rules_list

    def _format_apriori_rules(self):
        formatted_rules_text = []
        if not self.apriori_rules:
            formatted_rules_text.append(
                "No significant cross-selling rules found with the current settings."
            )
            return formatted_rules_text

        for i, rule in enumerate(self.apriori_rules):
            if i >= self.top_n_rules_output:
                break

            lhs_names = ", ".join(
                [self._get_product_display_name(pid) for pid in rule.lhs]
            )
            rhs_names = ", ".join(
                [self._get_product_display_name(pid) for pid in rule.rhs]
            )

            rule_text = (
                f"Customers who purchase ({lhs_names}) also frequently buy ({rhs_names}). "
                f"(Confidence: {rule.confidence * 100:.2f}%, Support: {rule.support * 100:.2f}%)"
            )
            formatted_rules_text.append(rule_text)

        if not formatted_rules_text:
            formatted_rules_text.append(
                "No cross-selling rules to display after formatting."
            )
        return formatted_rules_text

    def _generate_summaries(self):
        summaries = []
        if not self.pl_df.is_empty():
            total_value = 0
            if "value" in self.pl_df.columns and self.pl_df["value"].dtype.is_numeric():
                total_value = self.pl_df["value"].sum()

            unique_products = 0
            if self.item_id_col in self.pl_df.columns:
                unique_products = self.pl_df[self.item_id_col].n_unique()

            unique_transactions = 0
            if self.order_id_col in self.pl_df.columns:
                unique_transactions = self.pl_df[self.order_id_col].n_unique()

            summaries.append(
                f"Sales data analyzed for cross-selling involves {unique_products} unique items (using column '{self.item_id_col}') across {unique_transactions} transactions (using column '{self.order_id_col}'), with a total sales value of ${total_value:,.2f}."
            )
        else:
            summaries.append("No sales data available for cross-selling analysis.")

        summaries.append(
            f"Cross-selling analysis performed using the Apriori algorithm with settings: "
            f"minimum support={self.min_support * 100:.2f}%, minimum confidence={self.min_confidence * 100:.2f}%, max rule length={self.max_rule_length}."
        )

        num_rules_found = len(self.apriori_rules)
        num_rules_displayed = 0
        if self.rules_text:
            no_rules_messages = [
                "No significant cross-selling rules found with the current settings.",
                "No cross-selling rules to display after formatting.",
            ]
            # Check if the first rule text is not one of the "no rules" messages
            if self.rules_text[0] not in no_rules_messages:
                num_rules_displayed = len(self.rules_text)

        if num_rules_found > 0:
            summaries.append(
                f"Identified {num_rules_found} potential cross-selling association rules. Displaying top {num_rules_displayed} rules based on confidence and support."
            )
            summaries.append(
                "These rules highlight items or sets of items that are frequently purchased together, "
                "offering insights for bundling, recommendations, or targeted marketing."
            )
        else:
            summaries.append(
                "No significant cross-selling association rules were identified. "
                "This could be due to sparse purchasing patterns, restrictive Apriori settings, "
                f"or issues with the required data columns ('{self.order_id_col}', '{self.item_id_col}')."
            )
        return summaries
