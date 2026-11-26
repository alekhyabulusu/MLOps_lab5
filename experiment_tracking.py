import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Lab1-ElasticNet-Advanced"
MODEL_NAME = "Diabetes-ElasticNet"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def prepare_data():
    dataset = load_diabetes(as_frame=True)

    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def evaluate_metrics(y_true, predictions):
    predictions = np.asarray(predictions)

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    return {
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "mae": float(
            mean_absolute_error(y_true, predictions)
        ),
        "r2": float(
            r2_score(y_true, predictions)
        ),
    }


def create_mlflow_dataset(features, targets, name):
    dataframe = features.copy()
    dataframe["target"] = targets

    return mlflow.data.from_pandas(
        dataframe,
        name=name,
        targets="target",
    )


def run_experiments(
    tracking_uri,
    experiment_name,
    alphas,
    l1_ratios,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    x_train, x_test, y_train, y_test = prepare_data()

    train_dataset = create_mlflow_dataset(
        x_train, y_train, "diabetes_training"
    )

    test_dataset = create_mlflow_dataset(
        x_test, y_test, "diabetes_evaluation"
    )

    results = []

    with mlflow.start_run(
        run_name="ElasticNet-Hyperparameter-Sweep"
    ) as parent_run:

        mlflow.set_tags({
            "lab": "Lab1",
            "workflow": "hyperparameter_sweep",
            "model_family": "ElasticNet",
            "dataset": "sklearn_diabetes",
        })

        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "training_samples": len(x_train),
            "evaluation_samples": len(x_test),
            "number_of_trials": len(alphas) * len(l1_ratios),
            "alpha_values": str(alphas),
            "l1_ratio_values": str(l1_ratios),
        })

        mlflow.log_input(
            train_dataset, context="training"
        )

        mlflow.log_input(
            test_dataset, context="evaluation"
        )

        for alpha in alphas:
            for l1_ratio in l1_ratios:

                run_name = f"elasticnet_a{alpha}_l1{l1_ratio}"

                with mlflow.start_run(
                    run_name=run_name,
                    nested=True,
                ) as child_run:

                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                    )

                    model.fit(x_train, y_train)
                    predictions = model.predict(x_test)

                    metrics = evaluate_metrics(
                        y_test, predictions
                    )

                    mlflow.log_params({
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "random_state": RANDOM_STATE,
                        "max_iter": 10000,
                    })

                    mlflow.log_metrics(metrics)

                    mlflow.log_input(
                        train_dataset,
                        context="training",
                    )

                    mlflow.log_input(
                        test_dataset,
                        context="evaluation",
                    )

                    mlflow.set_tags({
                        "model_family": "ElasticNet",
                        "run_type": "hyperparameter_trial",
                    })

                    signature = infer_signature(
                        x_train, predictions
                    )

                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        signature=signature,
                        input_example=x_train.head(3),
                    )

                    result = {
                        "run_id": child_run.info.run_id,
                        "run_name": run_name,
                        "model_uri": model_info.model_uri,
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        **metrics,
                    }

                    results.append(result)

                    print(
                        f"Run: {run_name} | "
                        f"RMSE: {metrics['rmse']:.3f} | "
                        f"MAE: {metrics['mae']:.3f} | "
                        f"R2: {metrics['r2']:.3f}"
                    )

        # Select the best model from this sweep.
        best_run = min(
            results, key=lambda result: result["rmse"]
        )

        mlflow.log_metrics({
            "best_rmse": best_run["rmse"],
            "best_mae": best_run["mae"],
            "best_r2": best_run["r2"],
        })

        mlflow.log_params({
            "best_alpha": best_run["alpha"],
            "best_l1_ratio": best_run["l1_ratio"],
        })

        mlflow.set_tag(
            "best_run_id", best_run["run_id"]
        )

        # Register the best model.
        registered_model = mlflow.register_model(
            model_uri=best_run["model_uri"],
            name=MODEL_NAME,
        )

        # Assign the champion alias.
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=registered_model.version,
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=registered_model.version,
            key="selection_metric",
            value="rmse",
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=registered_model.version,
            key="rmse",
            value=str(best_run["rmse"]),
        )

        mlflow.set_tag(
            "registered_model_version",
            str(registered_model.version),
        )

        # Generate experiment reports.
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        summary = {
            "parent_run_id": parent_run.info.run_id,
            "experiment_name": experiment_name,
            "registered_model": MODEL_NAME,
            "model_version": str(registered_model.version),
            "model_alias": "champion",
            "best_run": best_run,
            "runs": results,
        }

        summary_path = output_dir / "run_summary.json"

        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        comparison_path = output_dir / "comparison.csv"

        pd.DataFrame(results).sort_values(
            "rmse"
        ).to_csv(
            comparison_path,
            index=False,
        )

        # Store reports in the parent MLflow run.
        mlflow.log_artifact(
            str(summary_path),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(comparison_path),
            artifact_path="reports",
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)

        print(f"Best run: {best_run['run_name']}")
        print(f"Best RMSE: {best_run['rmse']:.3f}")
        print(f"Best MAE: {best_run['mae']:.3f}")
        print(f"Best R2: {best_run['r2']:.3f}")
        print(f"Registered model: {MODEL_NAME}")
        print(f"Model version: {registered_model.version}")
        print("Model alias: champion")
        print(f"Summary saved to: {summary_path}")
        print(f"Comparison saved to: {comparison_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Advanced MLflow experiment tracking"
    )

    parser.add_argument(
        "--tracking-uri",
        default=TRACKING_URI,
    )

    parser.add_argument(
        "--experiment",
        default=EXPERIMENT_NAME,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1],
    )

    parser.add_argument(
        "--l1-ratios",
        nargs="+",
        type=float,
        default=[0.2, 0.8],
    )

    args = parser.parse_args()

    run_experiments(
        args.tracking_uri,
        args.experiment,
        args.alphas,
        args.l1_ratios,
    )


if __name__ == "__main__":
    main()
