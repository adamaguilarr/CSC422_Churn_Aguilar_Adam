# CSC422 Midterm - Customer Churn (Track 1)

Predict whether a telecom customer will churn using shallow learning models.

## Project Structure
```
CSC422_Churn_Aguilar_Adam/
  ├── data/                  # place raw CSV here or use Kaggle API to download
  ├── notebooks/             # EDA notebook(s)
  ├── src/                   # preprocessing, training, evaluation scripts
  ├── results/               # generated metrics and plots
  ├── presentation/          # slides or video outline
  ├── requirements.txt
  └── README.md
```

## Dataset
- **Telco Customer Churn** - IBM Sample Data (Kaggle).
- Kaggle URL: https://www.kaggle.com/blastchar/telco-customer-churn
- Expected filename: `data/Telco-Customer-Churn.csv`

### Getting the data
Option A - Kaggle CLI
1. Install Kaggle: `pip install kaggle`
2. Place your Kaggle API token `kaggle.json` in `~/.kaggle/`.
3. Run: `python data/download_telco.py`

Option B - Manual
1. Download the CSV from Kaggle in your browser.
2. Put the CSV at `data/Telco-Customer-Churn.csv`.

## Quickstart

1. Create a virtual environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Put the dataset at `data/Telco-Customer-Churn.csv` or run the downloader.

3. Run training and evaluation:
```
python src/train.py --data_path data/Telco-Customer-Churn.csv --out_dir results
```

4. Open the starter EDA notebook:
```
jupyter notebook notebooks/01_eda.ipynb
```

## Reproducibility
- Random seeds are set where applicable.
- Dependencies pinned in `requirements.txt`.

## Deliverables
- Baseline and model performance metrics in `results/`
- Key plots saved to `results/`
- Final summary added to the README and your video in `presentation/`

