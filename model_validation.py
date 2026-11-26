import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient

from experiment_tracking import (
    prepare_data,
    evaluate_metrics,
    create_mlflow_dataset,
    TRACKING_URI,
    MODEL_NAME,
)


VALIDATION_EXPERIMENT = "Lab1-Model-Validation"


def validate_model(
    tracking_uri=TRACKING_URI,
    model_name=MODEL_NAME,
    alias="champion",
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(VALIDATION_EXPERIMENT)

    client = MlflowClient()

    # Resolve the champion alias to a specific model version.
    model_version = client.get_model_version_by_alias(
        model_name,
        alias,
    )

    model_uri = (
        f"models:/{model_name}/{model_version.version}"
    )

    print(f"Loading model: {model_uri}")
    print(f"Resolved alias: {alias}")

    # Retrieve the original training run.
    original_run = client.get_run(
        model_version.run_id
    )

    original_metrics = {
        metric: float(original_run.data.metrics[metric])
        for metric in ["rmse", "mae", "r2"]
    }

    # Load the registered model without retraining.
    model = mlflow.sklearn.load_model(model_uri)

    _, x_test, _, y_test = prepare_data()

    with mlflow.start_run(
        run_name=f"validate_{model_name}_v{model_version.version}"
    ) as run:

        mlflow.log_params({
            "model_name": model_name,
            "model_version": model_version.version,
            "model_alias": alias,
            "evaluation_samples": len(x_test),
        })

        mlflow.set_tags({
            "workflow": "model_validation",
            "source_run_id": model_version.run_id,
        })

        evaluation_dataset = create_mlflow_dataset(
            x_test,
            y_test,
            "diabetes_validation",
        )

        mlflow.log_input(
            evaluation_dataset,
            context="validation",
        )

        # Generate predictions using the registered model.
        predictions = model.predict(x_test)

        metrics = evaluate_metrics(
            y_test,
            predictions,
        )

        mlflow.log_metrics(metrics)

        # Compare reloaded model metrics with original metrics.
        metric_checks = {}

        for metric, value in metrics.items():

            difference = abs(
                value - original_metrics[metric]
            )

            metric_checks[metric] = bool(
                np.isclose(
                    value,
                    original_metrics[metric],
                    rtol=1e-6,
                    atol=1e-8,
                )
            )

            mlflow.log_metric(
                f"{metric}_difference",
                difference,
            )

        validation_passed = (
            all(metric_checks.values())
            and len(predictions) == len(y_test)
            and bool(np.isfinite(predictions).all())
        )

        mlflow.log_metric(
            "validation_passed",
            int(validation_passed),
        )

        mlflow.set_tag(
            "validation_status",
            "PASSED" if validation_passed else "FAILED",
        )

        # Save predictions.
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        predictions_df = pd.DataFrame({
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "absolute_error": np.abs(
                y_test.to_numpy() - predictions
            ),
        })

        predictions_path = (
            output_dir / "validation_predictions.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        # Generate validation report.
        report = {
            "validation_run_id": run.info.run_id,
            "model_name": model_name,
            "model_version": str(model_version.version),
            "model_alias": alias,
            "source_run_id": model_version.run_id,
            "evaluation_samples": len(x_test),
            "original_metrics": original_metrics,
            "validation_metrics": metrics,
            "metric_checks": metric_checks,
            "validation_passed": validation_passed,
        }

        report_path = (
            output_dir / "validation_report.json"
        )

        report_path.write_text(
            json.dumps(report, indent=2) + "\n"
        )

        mlflow.log_artifact(
            str(predictions_path),
            artifact_path="validation",
        )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="validation",
        )

        print("\n" + "=" * 60)
        print("MODEL VALIDATION RESULTS")
        print("=" * 60)

        print(f"Model: {model_name}")
        print(f"Version: {model_version.version}")
        print(f"Alias: {alias}")
        print(f"Evaluation samples: {len(x_test)}")

        for metric, value in metrics.items():
            print(
                f"{metric.upper()}: {value:.3f} | "
                f"Original: {original_metrics[metric]:.3f}"
            )

        print(
            "Validation status:",
            "PASSED" if validation_passed else "FAILED",
        )

        print(f"Predictions saved to: {predictions_path}")
        print(f"Report saved to: {report_path}")

        if not validation_passed:
            raise RuntimeError(
                "Registered model failed validation."
            )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate a registered MLflow model"
    )

    parser.add_argument(
        "--tracking-uri",
        default=TRACKING_URI,
    )

    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--alias",
        default="champion",
    )

    args = parser.parse_args()

    validate_model(
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()