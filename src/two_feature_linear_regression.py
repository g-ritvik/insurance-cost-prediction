import csv
from pathlib import Path

from regression_models import (
    fit_normal_equation,
    fit_normal_equation_single_feature,
    fit_gradient_descent,
    predict,
    predict_baseline
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data" / "insurance.csv"


def load_data(csv_path: Path):
    print("[+] Loading insurance dataset...")

    bmi = []
    age = []
    expenses = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            bmi.append(float(row["bmi"]))
            age.append(float(row["age"]))
            expenses.append(float(row["expenses"]))

    print(f"[+] Loaded {len(expenses)} records")
    print("[+] Loaded features: bmi, age")
    print("[+] Loaded target: expenses")

    return bmi, age, expenses


def mse(y, predictions):
    return sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y, predictions)
    ) / len(y)


def rmse(y, predictions):
    return mse(y, predictions) ** 0.5


def mae(y, predictions):
    return sum(
        abs(actual - predicted)
        for actual, predicted in zip(y, predictions)
    ) / len(y)


def r2_score(y, predictions):
    y_mean = sum(y) / len(y)

    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y, predictions)
    )

    ss_tot = sum(
        (actual - y_mean) ** 2
        for actual in y
    )

    return 1 - (ss_res / ss_tot)


def save_results_csv(csv_path, results):
    print("[+] Saving model comparison results...")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "MSE",
            "RMSE",
            "MAE",
            "R2"
        ])

        for result in results:
            writer.writerow([
                result["Model"],
                result["MSE"],
                result["RMSE"],
                result["MAE"],
                result["R2"]
            ])

    print(f"[+] Results saved to: {csv_path}")


if __name__ == "__main__":
    bmi, age, expenses = load_data(PROJECT_DATA_DIR)

    baseline_weights = fit_normal_equation_single_feature(
        bmi,
        expenses
    )

    X = [
        [b, a]
        for b, a in zip(bmi, age)
    ]

    normal_weights = fit_normal_equation(
        X,
        expenses
    )

    gd_weights = fit_gradient_descent(
        X,
        expenses
    )

    baseline_predictions = predict_baseline(
        bmi,
        baseline_weights
    )

    normal_predictions = predict(
        X,
        normal_weights
    )

    gd_predictions = predict(
        X,
        gd_weights
    )

    baseline_metrics = {
        "MSE": mse(expenses, baseline_predictions),
        "RMSE": rmse(expenses, baseline_predictions),
        "MAE": mae(expenses, baseline_predictions),
        "R2": r2_score(expenses, baseline_predictions)
    }

    normal_metrics = {
        "MSE": mse(expenses, normal_predictions),
        "RMSE": rmse(expenses, normal_predictions),
        "MAE": mae(expenses, normal_predictions),
        "R2": r2_score(expenses, normal_predictions)
    }

    gd_metrics = {
        "MSE": mse(expenses, gd_predictions),
        "RMSE": rmse(expenses, gd_predictions),
        "MAE": mae(expenses, gd_predictions),
        "R2": r2_score(expenses, gd_predictions)
    }

    print("\n[+] Model Metrics")
    print(
        f"{'Model':<30} "
        f"{'MSE':>12} "
        f"{'RMSE':>12} "
        f"{'MAE':>12} "
        f"{'R2':>10}"
    )
    print("-" * 80)

    print(
        f"{'BMI-only Baseline':<30} "
        f"{baseline_metrics['MSE']:>12.2f} "
        f"{baseline_metrics['RMSE']:>12.2f} "
        f"{baseline_metrics['MAE']:>12.2f} "
        f"{baseline_metrics['R2']:>10.4f}"
    )

    print(
        f"{'Two-feature Normal Equation':<30} "
        f"{normal_metrics['MSE']:>12.2f} "
        f"{normal_metrics['RMSE']:>12.2f} "
        f"{normal_metrics['MAE']:>12.2f} "
        f"{normal_metrics['R2']:>10.4f}"
    )

    print(
        f"{'Two-feature Gradient Descent':<30} "
        f"{gd_metrics['MSE']:>12.2f} "
        f"{gd_metrics['RMSE']:>12.2f} "
        f"{gd_metrics['MAE']:>12.2f} "
        f"{gd_metrics['R2']:>10.4f}"
    )

    results = [
        {
            "Model": "BMI-only Baseline",
            **baseline_metrics
        },
        {
            "Model": "Two-feature Normal Equation",
            **normal_metrics
        },
        {
            "Model": "Two-feature Gradient Descent",
            **gd_metrics
        }
    ]

    RESULTS_PATH = PROJECT_ROOT / "reports" / "assignment_results.csv"

    save_results_csv(RESULTS_PATH, results)