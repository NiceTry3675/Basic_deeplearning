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
DIVERGENCE_THRESHOLD = 1e12
LR_SENSITIVITY_VALUES = [0.001, 0.01, 0.1]
EPOCH_SENSITIVITY_VALUES = [100, 1000, 5000]
SCALING_METHODS = ["zscore", "robust", "log_zscore"]
SCALING_METHOD_EPOCHS = [100, 5000]


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


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    scaling_method: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    train_features = train_df[feature_columns].to_numpy(dtype=float)
    test_features = test_df[feature_columns].to_numpy(dtype=float)

    if scaling_method == "zscore":
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        std[std == 0] = 1.0
        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std
    elif scaling_method == "log_zscore":
        if np.any(train_features < 0) or np.any(test_features < 0):
            raise ValueError("log_zscore requires non-negative feature values")
        train_features = np.log1p(train_features)
        test_features = np.log1p(test_features)
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        std[std == 0] = 1.0
        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std
    elif scaling_method == "robust":
        median = np.median(train_features, axis=0)
        q1 = np.percentile(train_features, 25, axis=0)
        q3 = np.percentile(train_features, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1.0
        train_features = (train_features - median) / iqr
        test_features = (test_features - median) / iqr
    elif scaling_method == "none":
        pass
    else:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")

    return train_features, test_features


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
) -> dict[str, object]:
    weights = np.zeros(features.shape[1], dtype=float)
    losses = []

    for epoch in range(epochs):
        gradient = compute_gradient(features, targets, weights)
        weights -= learning_rate * gradient

        with np.errstate(over="ignore", invalid="ignore"):
            loss = mse_loss(predict(features, weights), targets)

        losses.append(loss)

        if (not np.all(np.isfinite(weights))) or (not np.isfinite(loss)) or loss > DIVERGENCE_THRESHOLD:
            return {
                "weights": weights,
                "losses": losses,
                "status": "diverged",
                "message": f"Training diverged at epoch {epoch + 1}",
            }

    return {
        "weights": weights,
        "losses": losses,
        "status": "converged",
        "message": "Training converged",
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_sum_of_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_sum_of_squares = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def residual_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def padded_limits(values: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float]:
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    span = value_max - value_min
    padding = span * pad_ratio if span > 0 else max(abs(value_min) * pad_ratio, 1.0)
    return value_min - padding, value_max + padding


def zoom_window(losses: list[float], fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(losses, dtype=float)
    start = max(0, int(len(points) * (1.0 - fraction)))
    epochs = np.arange(len(points))
    return epochs[start:], points[start:]

def save_loss_curve(losses: list[float], case_name: str) -> None:
    epochs = np.arange(len(losses))
    zoom_epochs, zoom_losses = zoom_window(losses, fraction=0.2)
    final_loss = float(losses[-1])
    delta_losses = np.maximum(zoom_losses - final_loss, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(epochs, losses, color="tab:blue", linewidth=2)
    axes[0].set_title(f"{case_name} Loss (Full, Log Scale)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(zoom_epochs, delta_losses, color="tab:blue", linewidth=2)
    axes[1].set_title(f"{case_name} Loss Gap to Final (Last 20%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss - Final Loss")
    axes[1].set_xlim(zoom_epochs[0], zoom_epochs[-1] if len(zoom_epochs) > 1 else zoom_epochs[0] + 1)
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(FIGURE_DIR / f"{sanitize_name(case_name)}_loss.png")
    plt.close()


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, case_name: str) -> None:
    combined = np.concatenate([y_true, y_pred])
    full_low, full_high = padded_limits(combined)

    plt.figure(figsize=(6.8, 5))
    plt.scatter(y_true, y_pred, alpha=0.72, color="tab:green", edgecolors="white", linewidths=0.4)
    plt.plot([full_low, full_high], [full_low, full_high], linestyle="--", color="black")
    plt.title(f"{case_name} Prediction vs Actual")
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.xlim(full_low, full_high)
    plt.ylim(full_low, full_high)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{sanitize_name(case_name)}_prediction.png")
    plt.close()


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, case_name: str) -> None:
    residuals = y_true - y_pred
    x_low, x_high = padded_limits(y_pred)
    y_full_low, y_full_high = padded_limits(residuals)

    plt.figure(figsize=(6.8, 5))
    plt.scatter(y_pred, residuals, alpha=0.72, color="tab:red", edgecolors="white", linewidths=0.4)
    plt.axhline(0, linestyle="--", color="black")
    plt.title(f"{case_name} Residual Plot")
    plt.xlabel("Predicted MEDV")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.xlim(x_low, x_high)
    plt.ylim(y_full_low, y_full_high)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{sanitize_name(case_name)}_residual.png")
    plt.close()


def save_multi_loss_curve(
    curves: list[tuple[str, list[float]]],
    figure_name: str,
    title: str,
    show_zoom: bool = True,
) -> None:
    subplot_count = 2 if show_zoom else 1
    fig, axes = plt.subplots(1, subplot_count, figsize=(13, 4.8) if show_zoom else (6.8, 4.8))
    axes = np.atleast_1d(axes)
    relative_zoom_values = []

    for label, losses in curves:
        epochs = np.arange(len(losses))
        zoom_epochs, zoom_losses = zoom_window(losses, fraction=0.4)
        relative_epochs = np.linspace(0, 100, len(losses))
        zoom_relative_epochs = relative_epochs[max(0, int(len(losses) * 0.6)) :]
        final_loss = float(losses[-1])
        relative_gap = np.maximum(np.asarray(zoom_losses, dtype=float) - final_loss, 1e-12)
        axes[0].plot(epochs, losses, linewidth=2, label=label)
        if show_zoom:
            axes[1].plot(zoom_relative_epochs, relative_gap, linewidth=2, label=label)
        if show_zoom and len(relative_gap) > 0:
            relative_zoom_values.append(relative_gap)

    axes[0].set_title(f"{title} (Full, Log Scale)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if show_zoom:
        axes[1].set_title(f"{title} (Last 40%, Relative Progress)")
        axes[1].set_xlabel("Training Progress (%)")
        axes[1].set_ylabel("Loss - Final Loss")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        axes[1].set_xlim(60, 100)

        if relative_zoom_values:
            y_low, y_high = padded_limits(np.log10(np.concatenate(relative_zoom_values)))
            axes[1].set_ylim(10 ** y_low, 10 ** y_high)

    fig.tight_layout()
    plt.savefig(FIGURE_DIR / f"{sanitize_name(figure_name)}.png")
    plt.close()


def save_test_metric_plot(
    labels: list[str],
    mse_values: list[float],
    r2_values: list[float],
    figure_name: str,
    title: str,
) -> None:
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].bar(positions, mse_values, color="tab:blue", alpha=0.85)
    axes[0].set_title(f"{title} (Test MSE)")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    bar_colors = ["tab:green" if value >= 0 else "tab:red" for value in r2_values]
    axes[1].bar(positions, r2_values, color=bar_colors, alpha=0.85)
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title(f"{title} (Test R2)")
    axes[1].set_ylabel("Test R2")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.savefig(FIGURE_DIR / f"{sanitize_name(figure_name)}.png")
    plt.close()


def run_case(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    case_name: str,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    scaling_method: str = "zscore",
    save_plots: bool = True,
) -> dict[str, object]:
    x_train, x_test = prepare_features(
        train_df,
        test_df,
        feature_columns,
        scaling_method=scaling_method,
    )
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    x_train_bias = add_bias_term(x_train)
    x_test_bias = add_bias_term(x_test)

    training_result = train_gradient_descent(
        x_train_bias,
        y_train,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    result = {
        "case_name": case_name,
        "features": feature_columns,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "scaling_method": scaling_method,
        "weights": training_result["weights"],
        "losses": training_result["losses"],
        "status": training_result["status"],
        "message": training_result["message"],
    }

    if training_result["status"] != "converged":
        result["train_metrics"] = None
        result["test_metrics"] = None
        result["residual_stats"] = None
        return result

    train_predictions = predict(x_train_bias, training_result["weights"])
    test_predictions = predict(x_test_bias, training_result["weights"])

    result["train_metrics"] = regression_metrics(y_train, train_predictions)
    result["test_metrics"] = regression_metrics(y_test, test_predictions)
    result["residual_stats"] = residual_statistics(y_test, test_predictions)
    result["test_predictions"] = test_predictions

    if save_plots:
        save_loss_curve(training_result["losses"], case_name)
        save_prediction_plot(y_test, test_predictions, case_name)
        save_residual_plot(y_test, test_predictions, case_name)

    return result


def run_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    baseline_value = float(np.mean(y_train))

    train_predictions = np.full_like(y_train, baseline_value)
    test_predictions = np.full_like(y_test, baseline_value)

    return {
        "case_name": "Baseline",
        "baseline_value": baseline_value,
        "status": "fixed",
        "train_metrics": regression_metrics(y_train, train_predictions),
        "test_metrics": regression_metrics(y_test, test_predictions),
        "residual_stats": residual_statistics(y_test, test_predictions),
    }


def run_learning_rate_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    results = []
    curves = []

    for learning_rate in LR_SENSITIVITY_VALUES:
        result = run_case(
            train_df,
            test_df,
            feature_columns,
            case_name=f"LR {learning_rate}",
            learning_rate=learning_rate,
            epochs=EPOCHS,
            scaling_method="zscore",
            save_plots=False,
        )
        results.append(result)
        if result["losses"]:
            curves.append((f"lr={learning_rate}", result["losses"]))

    save_multi_loss_curve(curves, "lr_sensitivity_case_1", "Case 1 Learning Rate Sensitivity")
    save_test_metric_plot(
        labels=[f"lr={result['learning_rate']}" for result in results],
        mse_values=[result["test_metrics"]["mse"] for result in results],
        r2_values=[result["test_metrics"]["r2"] for result in results],
        figure_name="lr_sensitivity_test_metrics_case_1",
        title="Case 1 Learning Rate Sensitivity",
    )
    return results


def run_epoch_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    results = []
    curves = []

    for epochs in EPOCH_SENSITIVITY_VALUES:
        result = run_case(
            train_df,
            test_df,
            feature_columns,
            case_name=f"Epoch {epochs}",
            learning_rate=LEARNING_RATE,
            epochs=epochs,
            scaling_method="zscore",
            save_plots=False,
        )
        results.append(result)
        if result["losses"]:
            curves.append((f"epochs={epochs}", result["losses"]))

    save_multi_loss_curve(curves, "epoch_sensitivity_case_1", "Case 1 Epoch Sensitivity")
    save_test_metric_plot(
        labels=[f"ep={result['epochs']}" for result in results],
        mse_values=[result["test_metrics"]["mse"] for result in results],
        r2_values=[result["test_metrics"]["r2"] for result in results],
        figure_name="epoch_sensitivity_test_metrics_case_1",
        title="Case 1 Epoch Sensitivity",
    )
    return results


def run_standardization_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    results = []
    curves = []

    for use_standardization in [True, False]:
        label = "standardized" if use_standardization else "not_standardized"
        result = run_case(
            train_df,
            test_df,
            feature_columns,
            case_name=f"Standardization {label}",
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            scaling_method="zscore" if use_standardization else "none",
            save_plots=False,
        )
        results.append(result)
        if result["losses"]:
            curves.append((label, result["losses"]))

    save_multi_loss_curve(
        curves,
        "standardization_comparison_case_1",
        "Case 1 Standardization Comparison",
        show_zoom=False,
    )
    valid_results = [result for result in results if result["test_metrics"] is not None]
    save_test_metric_plot(
        labels=[result["scaling_method"] for result in valid_results],
        mse_values=[result["test_metrics"]["mse"] for result in valid_results],
        r2_values=[result["test_metrics"]["r2"] for result in valid_results],
        figure_name="standardization_test_metrics_case_1",
        title="Case 1 Standardization Comparison",
    )
    return results


def run_scaling_method_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    results = []
    curves = []

    for epochs in SCALING_METHOD_EPOCHS:
        for scaling_method in SCALING_METHODS:
            result = run_case(
                train_df,
                test_df,
                feature_columns,
                case_name=f"Scaling {scaling_method} epoch {epochs}",
                learning_rate=LEARNING_RATE,
                epochs=epochs,
                scaling_method=scaling_method,
                save_plots=False,
            )
            results.append(result)
            if result["losses"]:
                curves.append((f"{scaling_method} ({epochs})", result["losses"]))

    save_multi_loss_curve(
        curves,
        "scaling_method_comparison_case_1",
        "Case 1 Scaling Method Comparison",
    )
    save_test_metric_plot(
        labels=[f"{result['scaling_method']}\n{result['epochs']}" for result in results],
        mse_values=[result["test_metrics"]["mse"] for result in results],
        r2_values=[result["test_metrics"]["r2"] for result in results],
        figure_name="scaling_method_test_metrics_case_1",
        title="Case 1 Scaling Method Comparison",
    )
    return results


def print_metric_block(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}"
    )


def print_case_summary(case_result: dict[str, object]) -> None:
    print(f"\n[{case_result['case_name']}]")

    if case_result.get("features"):
        print(f"Features ({len(case_result['features'])}): {', '.join(case_result['features'])}")
        print(
            f"Learning rate: {case_result['learning_rate']}  "
            f"Epochs: {case_result['epochs']}  "
            f"Scaling: {case_result['scaling_method']}"
        )

    if case_result["status"] != "converged":
        print(f"Status: {case_result['status']} ({case_result['message']})")
        return

    print(f"Initial train loss: {case_result['losses'][0]:.4f}")
    print(f"Final train loss:   {case_result['losses'][-1]:.4f}")
    print_metric_block("Train metrics:", case_result["train_metrics"])
    print_metric_block("Test metrics: ", case_result["test_metrics"])

    residual_stats = case_result["residual_stats"]
    print(
        f"Residual stats: mean={residual_stats['mean']:.4f}  "
        f"std={residual_stats['std']:.4f}  "
        f"max_abs={residual_stats['max_abs']:.4f}"
    )


def print_baseline_summary(baseline_result: dict[str, object]) -> None:
    print(f"\n[{baseline_result['case_name']}]")
    print(f"Train mean MEDV: {baseline_result['baseline_value']:.4f}")
    print_metric_block("Train metrics:", baseline_result["train_metrics"])
    print_metric_block("Test metrics: ", baseline_result["test_metrics"])
    residual_stats = baseline_result["residual_stats"]
    print(
        f"Residual stats: mean={residual_stats['mean']:.4f}  "
        f"std={residual_stats['std']:.4f}  "
        f"max_abs={residual_stats['max_abs']:.4f}"
    )


def print_comparison_table(results: list[dict[str, object]], title: str) -> None:
    print(f"\n[{title}]")
    print(
        f"{'Model':<18} {'Status':<12} {'Train MSE':<12} "
        f"{'Test MSE':<12} {'Test R2':<10}"
    )

    for result in results:
        if result["train_metrics"] is None:
            train_mse = "-"
            test_mse = "-"
            test_r2 = "-"
        else:
            train_mse = f"{result['train_metrics']['mse']:.4f}"
            test_mse = f"{result['test_metrics']['mse']:.4f}"
            test_r2 = f"{result['test_metrics']['r2']:.4f}"

        print(
            f"{result['case_name']:<18} {result['status']:<12} {train_mse:<12} "
            f"{test_mse:<12} {test_r2:<10}"
        )


def print_sensitivity_table(
    title: str,
    results: list[dict[str, object]],
    config_label: str,
    config_key: str,
) -> None:
    print(f"\n[{title}]")
    print(
        f"{config_label:<14} {'Status':<12} {'Final Train Loss':<18} "
        f"{'Test MSE':<12} {'Test R2':<10}"
    )

    for result in results:
        if result["status"] == "converged":
            final_train_loss = f"{result['losses'][-1]:.4f}"
            test_mse = f"{result['test_metrics']['mse']:.4f}"
            test_r2 = f"{result['test_metrics']['r2']:.4f}"
        else:
            final_train_loss = "diverged"
            test_mse = "-"
            test_r2 = "-"

        print(
            f"{str(result[config_key]):<14} {result['status']:<12} {final_train_loss:<18} "
            f"{test_mse:<12} {test_r2:<10}"
        )


def print_scaling_method_table(results: list[dict[str, object]]) -> None:
    print("\n[Scaling Method Comparison]")
    print(
        f"{'Scaling':<12} {'Epochs':<8} {'Status':<12} {'Final Train Loss':<18} "
        f"{'Test MSE':<12} {'Test R2':<10}"
    )

    for result in results:
        if result["status"] == "converged":
            final_train_loss = f"{result['losses'][-1]:.4f}"
            test_mse = f"{result['test_metrics']['mse']:.4f}"
            test_r2 = f"{result['test_metrics']['r2']:.4f}"
        else:
            final_train_loss = "diverged"
            test_mse = "-"
            test_r2 = "-"

        print(
            f"{result['scaling_method']:<12} {result['epochs']:<8} {result['status']:<12} "
            f"{final_train_loss:<18} {test_mse:<12} {test_r2:<10}"
        )


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    train_df, test_df = split_dataset(df)

    all_features = get_feature_columns(df)
    correlations = get_feature_correlations(train_df)
    selected_features = select_top_correlated_features(train_df, limit=5)

    baseline_result = run_baseline(train_df, test_df)
    case_1_result = run_case(train_df, test_df, all_features, "Case 1")
    case_2_result = run_case(train_df, test_df, selected_features, "Case 2")

    lr_results = run_learning_rate_sensitivity(train_df, test_df, all_features)
    epoch_results = run_epoch_sensitivity(train_df, test_df, all_features)
    standardization_results = run_standardization_comparison(train_df, test_df, all_features)
    scaling_method_results = run_scaling_method_comparison(train_df, test_df, all_features)

    print("Boston Housing Linear Regression")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Excluded auxiliary column: {AUXILIARY_COLUMN}")
    print("\nTop correlated features on train split:")
    for feature_name, value in correlations.head(5).items():
        print(f"  {feature_name:<8} {value:.4f}")

    print_baseline_summary(baseline_result)
    print_case_summary(case_1_result)
    print_case_summary(case_2_result)
    print_comparison_table(
        [baseline_result, case_1_result, case_2_result],
        "Baseline / Main Model Comparison",
    )
    print_sensitivity_table("Learning Rate Sensitivity", lr_results, "Learning Rate", "learning_rate")
    print_sensitivity_table("Epoch Sensitivity", epoch_results, "Epochs", "epochs")
    print_sensitivity_table(
        "Standardization Comparison",
        standardization_results,
        "Standardized",
        "scaling_method",
    )
    print_scaling_method_table(scaling_method_results)

    print("\nSaved plots:")
    for figure_path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
