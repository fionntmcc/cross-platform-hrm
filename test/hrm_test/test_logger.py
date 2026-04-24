# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Unit Tests for hrm.training.logger

Covers:
    - TrainingLogger initialisation and file creation
    - JSONL writing
    - CSV writing
    - Record building (train + val merge)
    - Export to legacy JSON history
    - Plot generation (matplotlib soft-dependency)
    - Context manager protocol
    - TensorBoard disabled path
"""

import csv
import json

import pytest

from hrm.training.logger import TrainingLogger, _try_import_matplotlib

# Fixtures


@pytest.fixture()
def log_dir(tmp_path):
    """Provide a fresh temporary directory for each test."""
    return tmp_path / "logs"


@pytest.fixture()
def sample_train_metrics():
    return {
        "loss": 0.45,
        "lm_loss": 0.44,
        "token_accuracy": 0.87,
        "puzzle_accuracy": 0.30,
        "avg_residual": 0.0012,
        "num_samples": 900,
        "epoch_time_s": 12.3,
        "learning_rate": 0.001,
        "reasoning_steps": 16,
        "batch_losses": [0.5, 0.4, 0.45],  # list — should be excluded from flat record
    }


@pytest.fixture()
def sample_val_metrics():
    return {
        "loss": 0.50,
        "token_accuracy": 0.82,
        "puzzle_accuracy": 0.25,
        "avg_residual": 0.0015,
        "num_samples": 100,
    }


# Initialisation


class TestLoggerInit:
    """Logger creates expected directory and file handles."""

    def test_creates_log_dir(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="test", use_tensorboard=False)
        assert log_dir.exists()
        logger.close()

    def test_creates_jsonl_file(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="test", use_tensorboard=False)
        assert (log_dir / "test.jsonl").exists()
        logger.close()

    def test_creates_csv_file(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="test", use_tensorboard=False)
        assert (log_dir / "test.csv").exists()
        logger.close()

    def test_tb_disabled_no_tb_dir(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="test", use_tensorboard=False)
        assert not (log_dir / "tb").exists()
        logger.close()

    def test_default_run_name(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, use_tensorboard=False)
        assert logger.run_name == "training"
        logger.close()


# JSONL writing


class TestJSONLWriting:

    def test_single_epoch_written(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.close()

        lines = (log_dir / "r.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["epoch"] == 1
        assert record["train_loss"] == pytest.approx(0.45)

    def test_multiple_epochs_appended(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.log_epoch(2, sample_train_metrics)
        logger.close()

        lines = (log_dir / "r.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_val_metrics_included(self, log_dir, sample_train_metrics, sample_val_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics, sample_val_metrics)
        logger.close()

        record = json.loads((log_dir / "r.jsonl").read_text().strip())
        assert "val_loss" in record
        assert record["val_token_accuracy"] == pytest.approx(0.82)

    def test_list_values_excluded(self, log_dir, sample_train_metrics):
        """batch_losses (a list) should not appear in the flat record."""
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.close()

        record = json.loads((log_dir / "r.jsonl").read_text().strip())
        assert "train_batch_losses" not in record


# CSV writing


class TestCSVWriting:

    def test_csv_has_header(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.close()

        with open(log_dir / "r.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert "epoch" in rows[0]
        assert "train_loss" in rows[0]

    def test_csv_multiple_rows(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.log_epoch(2, sample_train_metrics)
        logger.close()

        with open(log_dir / "r.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_csv_values_correct(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.close()

        with open(log_dir / "r.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["train_loss"]) == pytest.approx(0.45)


# Record building


class TestBuildRecord:

    def test_prefixes_train_keys(self):
        record = TrainingLogger._build_record(
            epoch=1, train={"loss": 0.5, "token_accuracy": 0.9}, val=None
        )
        assert "train_loss" in record
        assert "train_token_accuracy" in record
        assert record["epoch"] == 1

    def test_prefixes_val_keys(self):
        record = TrainingLogger._build_record(epoch=1, train={"loss": 0.5}, val={"loss": 0.6})
        assert "val_loss" in record
        assert record["val_loss"] == 0.6

    def test_no_val(self):
        record = TrainingLogger._build_record(epoch=1, train={"loss": 0.5}, val=None)
        assert "val_loss" not in record

    def test_lists_excluded(self):
        record = TrainingLogger._build_record(
            epoch=1, train={"loss": 0.5, "batch_losses": [1, 2, 3]}, val=None
        )
        assert "train_batch_losses" not in record

    def test_dicts_excluded(self):
        record = TrainingLogger._build_record(
            epoch=1, train={"loss": 0.5, "nested": {"a": 1}}, val=None
        )
        assert "train_nested" not in record


# Export history JSON


class TestExportHistoryJSON:

    def test_export_creates_file(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.log_epoch(2, sample_train_metrics)
        path = logger.export_history_json()
        logger.close()

        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_export_legacy_format(self, log_dir, sample_train_metrics):
        """Exported JSON should have list-per-metric format."""
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        logger.log_epoch(2, sample_train_metrics)
        path = logger.export_history_json()
        logger.close()

        data = json.loads(path.read_text())
        assert "train_loss" in data
        assert isinstance(data["train_loss"], list)
        assert len(data["train_loss"]) == 2

    def test_export_custom_path(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        custom = log_dir / "custom_history.json"
        path = logger.export_history_json(path=custom)
        logger.close()

        assert path == custom
        assert custom.exists()


# Plot generation


class TestPlotGeneration:

    def test_no_data_returns_none(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        result = logger.plot_training_curves()
        logger.close()
        assert result is None

    @pytest.mark.skipif(
        _try_import_matplotlib() is None,
        reason="matplotlib not installed",
    )
    def test_plot_creates_png(self, log_dir, sample_train_metrics, sample_val_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics, sample_val_metrics)
        logger.log_epoch(2, sample_train_metrics, sample_val_metrics)
        path = logger.plot_training_curves()
        logger.close()

        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    @pytest.mark.skipif(
        _try_import_matplotlib() is None,
        reason="matplotlib not installed",
    )
    def test_plot_custom_path(self, log_dir, sample_train_metrics):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        custom = log_dir / "my_curves.png"
        path = logger.plot_training_curves(save_path=custom)
        logger.close()

        assert path == custom
        assert custom.exists()

    @pytest.mark.skipif(
        _try_import_matplotlib() is None,
        reason="matplotlib not installed",
    )
    def test_plot_train_only(self, log_dir, sample_train_metrics):
        """Plot should work without val metrics."""
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        logger.log_epoch(1, sample_train_metrics)
        path = logger.plot_training_curves()
        logger.close()
        assert path is not None


# Context manager


class TestContextManager:

    def test_enter_returns_self(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        with logger as ctx:
            assert ctx is logger

    def test_files_closed_after_exit(self, log_dir, sample_train_metrics):
        with TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False) as logger:
            logger.log_epoch(1, sample_train_metrics)
        assert logger._jsonl_file.closed
        assert logger._csv_file.closed


# log_epoch return value


class TestLogEpochReturn:

    def test_returns_dict(self, log_dir, sample_train_metrics):
        with TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False) as logger:
            record = logger.log_epoch(1, sample_train_metrics)
        assert isinstance(record, dict)
        assert record["epoch"] == 1

    def test_history_populated(self, log_dir, sample_train_metrics):
        with TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False) as logger:
            logger.log_epoch(1, sample_train_metrics)
            logger.log_epoch(2, sample_train_metrics)
            assert len(logger._history) == 2


# TensorBoard disabled path


class TestTensorBoardDisabled:

    def test_no_tb_writer(self, log_dir):
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        assert logger._tb_writer is None
        logger.close()

    def test_log_epoch_without_tb(self, log_dir, sample_train_metrics):
        """Logging should work fine without TensorBoard."""
        logger = TrainingLogger(log_dir=log_dir, run_name="r", use_tensorboard=False)
        record = logger.log_epoch(1, sample_train_metrics)
        assert record["epoch"] == 1
        logger.close()
