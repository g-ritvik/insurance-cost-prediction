# Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-orange)

A two-feature linear regression project using `bmi` and `age` to predict `expenses`. The project implements both the Normal Equation and Gradient Descent from scratch and compares them with a BMI-only baseline.

## Project Structure

```text
insurance-cost-prediction/
├── data/
│   ├── insurance.csv
│   └── data_download.py
├── reports/
│   ├── assignment_results.csv
│   └── MEMO.md
├── src/
│   ├── regression_models.py
│   └── two_feature_linear_regression.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

* Python 3
* `kagglehub` — only required if the dataset needs to be downloaded again

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/g-ritvik/insurance-cost-prediction.git
cd insurance-cost-prediction
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required package

```bash
python -m pip install -r requirements.txt
```

On macOS/Linux, use:

```bash
python3 -m pip install -r requirements.txt
```

## Dataset

The repository already includes:

```text
data/insurance.csv
```

No download is required to run the analysis.

### Download the Dataset Again

If `insurance.csv` is missing or needs to be downloaded again, run:

**Windows:**

```powershell
python data/data_download.py
```

**macOS/Linux:**

```bash
python3 data/data_download.py
```

The download script uses `kagglehub` to retrieve the dataset.

## Run the Analysis

From the project root, run:

**Windows:**

```powershell
python src/two_feature_linear_regression.py
```

**macOS/Linux:**

```bash
python3 src/two_feature_linear_regression.py
```

The script:

1. Loads `bmi`, `age`, and `expenses`.
2. Fits a BMI-only baseline.
3. Fits a two-feature Normal Equation model.
4. Fits a two-feature Gradient Descent model.
5. Calculates MSE, RMSE, MAE, and R².
6. Prints a model comparison table.
7. Saves the results to `reports/assignment_results.csv`.

## Models

The project compares:

* **BMI-only baseline**
* **Two-feature Normal Equation**
* **Two-feature Gradient Descent**

The two-feature models use:

```text
expenses = w0 + w1 × bmi + w2 × age
```

Gradient Descent standardizes the features during training and converts the resulting coefficients back to the original feature scale.

## Results

The models produced the following results:

| Model                        |          MSE |     RMSE |     MAE |     R² |
| ---------------------------- | -----------: | -------: | ------: | -----: |
| BMI-only Baseline            | 140764214.67 | 11864.41 | 9172.30 | 0.0394 |
| Two-feature Normal Equation  | 129359773.29 | 11373.64 | 9032.28 | 0.1173 |
| Two-feature Gradient Descent | 129359773.29 | 11373.64 | 9032.28 | 0.1173 |

The Normal Equation and Gradient Descent produced the same coefficients to two decimal places.

## Reports

* `reports/assignment_results.csv` — model metrics and comparison
* `reports/MEMO.md` — interpretation of the results and model findings

## Troubleshooting

### `ModuleNotFoundError: No module named 'kagglehub'`

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### `FileNotFoundError: data/insurance.csv`

Download the dataset again:

```bash
python data/data_download.py
```

### Python command not recognized

Make sure Python 3 is installed and available from your terminal. On some systems, use `python3` instead of `python`.
