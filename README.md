# Insurance Cost Prediction

A Python linear regression project using real insurance data to examine how `bmi` and `age` relate to `expenses`.

The project compares a BMI-only baseline with a two-feature model trained using the Normal Equation and Gradient Descent.

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
├── .gitignore
└── README.md
```

### Main Files

* `data/insurance.csv` — dataset used for the analysis
* `data/data_download.py` — utility for downloading the dataset
* `src/regression_models.py` — regression, matrix-solving, standardization, and prediction functions
* `src/two_feature_linear_regression.py` — main analysis script
* `reports/assignment_results.csv` — model comparison results
* `reports/MEMO.md` — interpretation of the results

## Requirements

* Python 3.10 or newer
* Git
* Internet access for cloning the repository

No additional Python packages are required to run the main analysis.

## Setup

### 1. Clone the repository

#### Windows

Open PowerShell:

```powershell
git clone https://github.com/g-ritvik/insurance-cost-prediction.git
cd insurance-cost-prediction
```

#### macOS

Open Terminal:

```bash
git clone https://github.com/g-ritvik/insurance-cost-prediction.git
cd insurance-cost-prediction
```

### 2. Check Python

#### Windows

```powershell
python --version
```

If needed:

```powershell
py --version
```

#### macOS

```bash
python3 --version
```

Python 3.10 or newer is recommended.

### 3. Verify the dataset

The dataset is already included at:

```text
data/insurance.csv
```

No additional download is required to run the analysis.

The repository also includes `data/data_download.py` if the dataset needs to be downloaded again.

## Run the Analysis

### Windows

```powershell
python src/two_feature_linear_regression.py
```

If needed:

```powershell
py src/two_feature_linear_regression.py
```

### macOS

```bash
python3 src/two_feature_linear_regression.py
```

The script loads the data, trains the models, calculates the evaluation metrics, prints a comparison table, and saves the results to:

```text
reports/assignment_results.csv
```

## Models

### BMI-only Baseline

```text
expenses = w0 + w1 * bmi
```

Used as the baseline for comparison.

### Two-Feature Normal Equation

```text
expenses = w0 + w1 * bmi + w2 * age
```

The Normal Equation is solved using a general Gauss-Jordan matrix method.

### Two-Feature Gradient Descent

The `bmi` and `age` features are standardized before training. The resulting weights are then converted back to their original feature units.

## Results

Expected results are approximately:

```text
Model                                   MSE         RMSE          MAE         R2
--------------------------------------------------------------------------------
BMI-only Baseline              140764214.67     11864.41      9172.30     0.0394
Two-feature Normal Equation    129359773.29     11373.64      9032.28     0.1173
Two-feature Gradient Descent   129359773.29     11373.64      9032.28     0.1173
```

The two-feature models produce approximately:

```text
w0 = -6437.35
w1 = 333.39
w2 = 241.90
```

Adding `age` increases R² from `0.0394` to `0.1173`. The Normal Equation and Gradient Descent produce the same weights to two decimal places.

The full interpretation is available in `reports/MEMO.md`.

## Troubleshooting

**Python is not recognized**

* Windows: try `py --version`
* macOS: try `python3 --version`

**Dataset not found**

Make sure `data/insurance.csv` exists and that you are running the command from the `insurance-cost-prediction` folder.

**Results differ slightly**

Small differences in decimal places can occur because of floating-point calculations.
