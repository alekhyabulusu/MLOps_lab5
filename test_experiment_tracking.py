import unittest
import tempfile
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from experiment_tracking import (
    prepare_data,
    evaluate_metrics,
    create_mlflow_dataset,
    run_experiments,
    MODEL_NAME,
)
from model_validation import validate_model


class ExperimentTrackingTests(unittest.TestCase):

    def test_metrics_perfect_predictions(self):
        actual = np.array([10, 20, 30, 40])
        predicted = np.array([10, 20, 30, 40])

        metrics = evaluate_metrics(actual, predicted)

        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_metrics_imperfect_predictions(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([12, 18, 32])

        metrics = evaluate_metrics(actual, predicted)

        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertLess(metrics["r2"], 1.0)

    def test_split_reproducible(self):
        split_one = prepare_data()
        split_two = prepare_data()

        for first, second in zip(split_one, split_two):
            if isinstance(first, pd.DataFrame):
                pd.testing.assert_frame_equal(first, second)
            else:
                pd.testing.assert_series_equal(first, second)

    def test_dataset_dimensions(self):
        x_train, x_test, y_train, y_test = prepare_data()

        self.assertEqual(len(x_train), 331)
        self.assertEqual(len(x_test), 111)

        self.assertEqual(x_train.shape[1], 10)
        self.assertEqual(x_test.shape[1], 10)

        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))

    def test_no_train_test_overlap(self):
        x_train, x_test, _, _ = prepare_data()

        overlap = set(x_train.index).intersection(
            set(x_test.index)
        )

        self.assertEqual(len(overlap), 0)

    def test_reject_nan_predictions(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([10, np.nan, 30])

        with self.assertRaises(ValueError):
            evaluate_metrics(actual, predicted)

    def test_reject_infinite_predictions(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([10, np.inf, 30])

        with self.assertRaises(ValueError):
            evaluate_metrics(actual, predicted)

    def test_mlflow_dataset_lineage(self):
        x_train, _, y_train, _ = prepare_data()

        dataset = create_mlflow_dataset(
            x_train,
            y_train,
            "test_training_dataset",
        )

        self.assertEqual(
            dataset.name,
            "test_training_dataset",
        )

        self.assertIn(
            "target",
            dataset.df.columns,
        )

        self.assertEqual(
            len(dataset.df),
            len(x_train),
        )

        self.assertTrue(dataset.digest)

        # Verify that the original features were not modified.
        self.assertNotIn("target", x_train.columns)

        np.testing.assert_array_equal(
            dataset.df["target"].to_numpy(),
            y_train.to_numpy(),
        )


class MLflowIntegrationTests(unittest.TestCase):

    def test_complete_mlflow_lifecycle(self):

        previous_tracking_uri = mlflow.get_tracking_uri()
        previous_registry_uri = mlflow.get_registry_uri()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                temp_path = Path(temp_dir)

                tracking_uri = (
                    f"sqlite:///{temp_path / 'test_mlflow.db'}"
                )

                experiment_name = "Test-ElasticNet-Lifecycle"

                # Redirect generated reports and artifacts to
                # temporary storage, preserving real lab results.
                original_cwd = Path.cwd()

                import os

                try:
                    os.chdir(temp_path)

                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_registry_uri(tracking_uri)

                    client = MlflowClient()

                    client.create_experiment(
                        experiment_name,
                        artifact_location=(
                            temp_path / "artifacts"
                        ).as_uri(),
                    )

                    client.create_experiment(
                        "Lab1-Model-Validation",
                        artifact_location=(
                            temp_path / "validation_artifacts"
                        ).as_uri(),
                    )

                    # Use two configurations for a faster test.
                    summary = run_experiments(
                        tracking_uri=tracking_uri,
                        experiment_name=experiment_name,
                        alphas=[0.001, 0.01],
                        l1_ratios=[0.8],
                    )

                    self.assertEqual(len(summary["runs"]), 2)

                    parent_id = summary["parent_run_id"]

                    parent_run = client.get_run(parent_id)

                    self.assertEqual(
                        parent_run.info.status,
                        "FINISHED",
                    )

                    experiment = (
                        client.get_experiment_by_name(
                            experiment_name
                        )
                    )

                    child_runs = client.search_runs(
                        experiment_ids=[
                            experiment.experiment_id
                        ],
                        filter_string=(
                            f"tags.mlflow.parentRunId = "
                            f"'{parent_id}'"
                        ),
                    )

                    self.assertEqual(len(child_runs), 2)

                    for run in child_runs:
                        self.assertEqual(
                            run.info.status,
                            "FINISHED",
                        )

                        self.assertIn(
                            "rmse",
                            run.data.metrics,
                        )

                        self.assertIn(
                            "alpha",
                            run.data.params,
                        )

                        self.assertEqual(
                            len(run.inputs.dataset_inputs),
                            2,
                        )

                    # Verify best-model selection.
                    expected_rmse = min(
                        run["rmse"]
                        for run in summary["runs"]
                    )

                    self.assertAlmostEqual(
                        summary["best_run"]["rmse"],
                        expected_rmse,
                    )

                    # Verify registered champion.
                    champion = (
                        client.get_model_version_by_alias(
                            MODEL_NAME,
                            "champion",
                        )
                    )

                    self.assertEqual(
                        str(champion.version),
                        summary["model_version"],
                    )

                    self.assertEqual(
                        champion.run_id,
                        summary["best_run"]["run_id"],
                    )

                    # Verify report artifacts.
                    artifacts = client.list_artifacts(
                        parent_id,
                        "reports",
                    )

                    artifact_paths = {
                        artifact.path
                        for artifact in artifacts
                    }

                    self.assertIn(
                        "reports/run_summary.json",
                        artifact_paths,
                    )

                    self.assertIn(
                        "reports/comparison.csv",
                        artifact_paths,
                    )

                    # Validate the registered model.
                    report = validate_model(
                        tracking_uri=tracking_uri,
                        model_name=MODEL_NAME,
                        alias="champion",
                    )

                    self.assertTrue(
                        report["validation_passed"]
                    )

                    self.assertTrue(
                        all(report["metric_checks"].values())
                    )

                    self.assertEqual(
                        report["evaluation_samples"],
                        111,
                    )

                finally:
                    os.chdir(original_cwd)

        finally:
            mlflow.set_tracking_uri(previous_tracking_uri)
            mlflow.set_registry_uri(previous_registry_uri)


if __name__ == "__main__":
    unittest.main()