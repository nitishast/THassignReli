# New Relic Data Science Challenge

## Project Overview
This project analyzes S&P 500 stocks to identify groups of stocks that move in sync, regardless of their sectors.

## Project Structure
```
NEWRELIC2/
│
├── data/
│   ├── feature_engineering.py
│   ├── main.py
│   ├── prepared_data_2024-10-03.csv
│   ├── sector_correlation.csv
│   └── training_data.csv
│
├── model/
│   ├── best_model.pkl
│   └── metrics.txt
│
├── notebook/
│   ├── clustering.ipynb
│   ├── corr.csv
│   ├── dataexploration.ipynb
│   ├── df_training.csv
│   ├── feature_engineering.ipynb
│   ├── feature_engineering.py
│   ├── final_df.csv
│   ├── label_encoder.pkl
│   ├── labeled_data_updated_2_year_fixed.csv
│   ├── labeled_data_updated_2_year.csv
│   ├── modular.ipynb
│   ├── okay, done.csv
│   ├── prepared_data.csv
│   ├── training.ipynb
│   ├── training.py
│   └── y.csv
│
├── public/
│   └── public_data(2024-10-03).csv
│
├── src/
│   ├── labelling.py
│   ├── model_dispatcher.py
│   ├── train_optimized.py
│   └── train.py
│
├── .gitignore
├── config.py
├── dockerfile
├── homepage.py
└── requirments.txt
```

## Pipeline Overview
1. Data preparation and exploration
2. Feature engineering
3. Stock labeling
4. Model training and optimization
5. Clustering analysis

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `streamlit run homepage.py` to execute the pipeline and start the UI

## Usage
- Use notebooks in the `notebook/` directory for exploratory analysis
- Execute `src/train.py` or `src/train_optimized.py` for model training
- Refer to `model/metrics.txt` for performance evaluation

## Docker
A Dockerfile is provided for containerization.