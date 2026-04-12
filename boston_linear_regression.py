from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("BostonHousing.csv")
FIGURE_DIR = Path("figures")
TARGET_COLUMN = "MEDV"
AUXILIARY_COLUMN = "CAT. MEDV"
TRAIN_SIZE = 400
LEARNING_RATE = 0.01
EPOCHS = 5000


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) != 506:
        raise ValueError(f"Expected 506 rows, found {len(df)}")
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[:TRAIN_SIZE].copy()
    test_df = df.iloc[TRAIN_SIZE:].copy()
    if len(train_df) != 400 or len(test_df) != 106:
        raise ValueError("Dataset split must be 400 train rows and 106 test rows")
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, AUXILIARY_COLUMN}
    return [column for column in df.columns if column not in excluded]


def get_feature_correlations(train_df: pd.DataFrame) -> pd.Series:
    feature_columns = get_feature_columns(train_df)
    correlations = train_df[feature_columns + [TARGET_COLUMN]].corr(numeric_only=True)[TARGET_COLUMN]
    return correlations.drop(labels=[TARGET_COLUMN]).sort_values(key=lambda s: s.abs(), ascending=False)


def select_top_correlated_features(train_df: pd.DataFrame, limit: int = 5) -> list[str]:
    return get_feature_correlations(train_df).head(limit).index.tolist()


def standardize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = train_df[feature_columns].to_numpy(dtype=float)
    test_features = test_df[feature_columns].to_numpy(dtype=float)

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0

    x_train = (train_features - mean) / std
    x_test = (test_features - mean) / std
    return x_train, x_test, mean, std


def add_bias_term(features: np.ndarray) -> np.ndarray:
    bias = np.ones((features.shape[0], 1))
    return np.hstack([bias, features])


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features @ weights


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    residuals = predictions - targets
    return float(np.mean(residuals ** 2))


def compute_gradient(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    residuals = predict(features, weights) - targets
    return (2 / len(features)) * (features.T @ residuals)


def train_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, list[float]]:
    weights = np.zeros(features.shape[1], dtype=float)
    losses = []

    for epoch in range(epochs):
        gradient = compute_gradient(features, targets, weights)
        weights -= learning_rate * gradient
        losses.append(mse_loss(predict(features, weights), targets))

        if not np.isfinite(losses[-1]):
            raise FloatingPointError("Training diverged. Check the learning rate or feature scaling.")

    return weights, losses


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_sum_of_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_sum_of_squares = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def save_loss_curve(losses: list[float], case_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color="tab:blue", linewidth=2)
    plt.title(f"{case_name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{case_name.lower().replace(' ', '_')}_loss.png")
    plt.close()


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, case_name: str) -> None:
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="tab:green")
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.title(f"{case_name} Prediction vs Actual")
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{case_name.lower().replace(' ', '_')}_prediction.png")
    plt.close()


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, case_name: str) -> None:
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7, color="tab:red")
    plt.axhline(0, linestyle="--", color="black")
    plt.title(f"{case_name} Residual Plot")
    plt.xlabel("Predicted MEDV")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{case_name.lower().replace(' ', '_')}_residual.png")
    plt.close()


def run_case(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    case_name: str,
    save_plots: bool = True,
) -> dict[str, object]:
    x_train, x_test, _, _ = standardize_features(train_df, test_df, feature_columns)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    x_train_bias = add_bias_term(x_train)
    x_test_bias = add_bias_term(x_test)

    weights, losses = train_gradient_descent(
        x_train_bias,
        y_train,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    train_predictions = predict(x_train_bias, weights)
    test_predictions = predict(x_test_bias, weights)

    train_metrics = regression_metrics(y_train, train_predictions)
    test_metrics = regression_metrics(y_test, test_predictions)

    if save_plots:
        save_loss_curve(losses, case_name)
        save_prediction_plot(y_test, test_predictions, case_name)
        save_residual_plot(y_test, test_predictions, case_name)

    return {
        "case_name": case_name,
        "features": feature_columns,
        "weights": weights,
        "losses": losses,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_predictions,
    }


def print_case_summary(case_result: dict[str, object]) -> None:
    case_name = case_result["case_name"]
    features = case_result["features"]
    losses = case_result["losses"]
    train_metrics = case_result["train_metrics"]
    test_metrics = case_result["test_metrics"]

    print(f"\n[{case_name}]")
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Initial train loss: {losses[0]:.4f}")
    print(f"Final train loss:   {losses[-1]:.4f}")
    print("Train metrics:")
    print(
        f"  MSE={train_metrics['mse']:.4f}  RMSE={train_metrics['rmse']:.4f}  "
        f"MAE={train_metrics['mae']:.4f}  R2={train_metrics['r2']:.4f}"
    )
    print("Test metrics:")
    print(
        f"  MSE={test_metrics['mse']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
        f"MAE={test_metrics['mae']:.4f}  R2={test_metrics['r2']:.4f}"
    )


def print_comparison(case_1_result: dict[str, object], case_2_result: dict[str, object]) -> None:
    print("\n[Case Comparison]")
    print(
        f"{'Case':<10} {'Feature Count':<14} {'Test MSE':<12} "
        f"{'Test RMSE':<12} {'Test MAE':<12} {'Test R2':<10}"
    )
    for result in [case_1_result, case_2_result]:
        metrics = result["test_metrics"]
        print(
            f"{result['case_name']:<10} {len(result['features']):<14} "
            f"{metrics['mse']:<12.4f} {metrics['rmse']:<12.4f} "
            f"{metrics['mae']:<12.4f} {metrics['r2']:<10.4f}"
        )


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    train_df, test_df = split_dataset(df)

    all_features = get_feature_columns(df)
    correlations = get_feature_correlations(train_df)
    selected_features = select_top_correlated_features(train_df, limit=5)

    case_1_result = run_case(train_df, test_df, all_features, "Case 1")
    case_2_result = run_case(train_df, test_df, selected_features, "Case 2")

    print("Boston Housing Linear Regression")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Excluded auxiliary column: {AUXILIARY_COLUMN}")
    print("\nTop correlated features on train split:")
    for feature_name, value in correlations.head(5).items():
        print(f"  {feature_name:<8} {value:.4f}")

    print_case_summary(case_1_result)
    print_case_summary(case_2_result)
    print_comparison(case_1_result, case_2_result)

    print("\nSaved plots:")
    for figure_path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
