# 🧠 Sales_Chat-BI_bot

A **Streamlit-based Chatbot** that allows you to interact with sales data using natural language. Built with modular AI contexts for forecasting, cross-selling, and brand-region analysis.

> ⚡ Ask questions like:
> - *"Plot sales for INDOMIE PULL in NORTH 1 last month"*
> - *"Forecast quantity for COLGATE in EAST branch"*
> - *"Show top 3 brands in SOUTH region by revenue"*

---

## 🗂️ Project Structure

Sales_Chat-BI_bot/
│
├── cbot/
│ ├── app_model_context2.py # Streamlit app entry point
│ ├── chatbot_model_context3.py # Core LLM interface logic
│ ├── cross_sell_model_context.py # Context logic for cross-sell suggestions
│ ├── region_model_context.py # Regional insights
│ ├── top_model_context.py # Top-selling brands/products
│ ├── region_normalizer.py # Preprocessing helper
│ ├── sales_forecaster.py # Time series forecasting logic
│ ├── utils_helpers_model_context.py # Shared utilities
│ ├── sales_rt.csv # 🔽 (Download separately)
│ ├── sales_rt.parquet # 🔽 (Download separately)
│ ├── dummy_sales_rt.parquet # (Light sample for testing)
│ ├── metayb-logo.png # Logo used in sidebar
│ ├── requirements.txt
│ └── .gitignore
│
├── cbot_backup/ # Backup of important files
└── README.md

Sales_Chat-BI_bot/
├── cbot/
│   ├── app_model_context2.py         # Streamlit app entry point
│   ├── chatbot_model_context3.py     # Core LLM interface logic
│   ├── cross_sell_model_context.py   # Context logic for cross-sell suggestions
│   ├── region_model_context.py       # Regional insights
│   ├── top_model_context.py          # Top-selling brands/products
│   ├── region_normalizer.py          # Preprocessing helper
│   ├── sales_forecaster.py           # Time series forecasting logic
│   ├── utils_helpers_model_context.py# Shared utilities
│   ├── sales_rt.csv                  # 🔽 (Download separately)
│   ├── sales_rt.parquet              # 🔽 (Download separately)
│   ├── dummy_sales_rt.parquet        # ✅ (Light sample for testing)
│   ├── metayb-logo.png               # Logo used in sidebar
│   └── requirements.txt              # Project dependencies
├── cbot_backup/                      # Backup of important files
└── README.md                         # Project documentation



---

## 📥 Download Required Data Files

This project uses large datasets which are not included in GitHub due to file size limits.

📦 Download the following files manually from this [Google Drive folder](https://drive.google.com/drive/folders/16M1jhAAlE9HgTqVnlYfDA69djVOIwVE7?usp=sharing):

- `sales_rt.csv` (~425MB)
- `sales_rt.parquet` (~52MB)

> After downloading, place them inside the `cbot/` directory.

---

## ⚙️ Setup Instructions

Follow the steps below to run the chatbot locally:

### 1 Clone the Repository

```bash
git clone https://github.com/subhaorku/Sales_Chat-BI_bot.git
cd Sales_Chat-BI_bot/cbot
```
### 2 Set Up a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Mac/Linux

```
### 3 Install Dependencies
```bash
pip install -r requirements.txt
```
### 4 Place the Data Files

cbot/
├── sales_rt.csv
└── sales_rt.parquet


### 5 Run the Streamlit Chatbot
From inside the cbot/ directory:
```bash
streamlit run app_model_context2.py
Then visit: http://localhost:8501

```
### Example Prompts
"Show trend for INDOMIE PULL in NORTH 1"

"Forecast sales for DANO next 3 weeks"

"What are the top 3 brands in EAST last month?"

"Compare sales of COLGATE and DANO in SOUTH"
