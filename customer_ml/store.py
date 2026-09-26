"""Load a trusted local Joblib model or train and persist it once per CSV/ZIP."""

import hashlib
import os
from pathlib import Path
import pickle
import tempfile

import joblib
import sklearn

from .data import DATA_PATH, ROOT, CustomerDataset, read_dataset
from .model import CustomerPredictor
from .train import evaluate

CACHE_VERSION = 4
ARTIFACT_DIR = ROOT / "artifacts" / "customer_ml"


def _cache_path(contents: bytes, filename: str) -> Path:
    if filename == DATA_PATH.name and hashlib.sha256(contents).digest() == hashlib.sha256(DATA_PATH.read_bytes()).digest():
        return ARTIFACT_DIR / "model.joblib"
    digest = hashlib.sha256(contents).hexdigest()
    return ARTIFACT_DIR / "cache" / f"{digest}.joblib"


def _load_bundle(path: Path, contents: bytes, filename: str):
    if not path.is_file():
        return None
    try:
        bundle = joblib.load(path)
        dataset = bundle["dataset"]
        predictor = bundle["predictor"]
        report = bundle["report"]
        if (bundle["version"] != CACHE_VERSION or not isinstance(dataset, CustomerDataset)
                or not isinstance(predictor, CustomerPredictor)
                or dataset.source_sha256 != hashlib.sha256(contents).hexdigest()
                or dataset.source_name != Path(filename).name
                or report["environment"]["scikit_learn"] != sklearn.__version__
                or predictor.model_params != report["selected_params"]):
            return None
        return dataset, predictor, report
    except (OSError, EOFError, ValueError, KeyError, AttributeError, TypeError,
            ImportError, IndexError, pickle.UnpicklingError):
        return None


def load_or_train(contents: bytes, filename: str):
    """Reuse the fitted model and its holdout report across server restarts."""
    path = _cache_path(contents, filename)
    cached = _load_bundle(path, contents, filename)
    if cached is not None:
        return cached
    dataset = read_dataset(contents, filename)
    report, _ = evaluate(dataset)
    predictor = CustomerPredictor(dataset.data, report["selected_params"])
    report["final_training_rows"] = len(dataset.data)
    bundle = {"version": CACHE_VERSION, "dataset": dataset, "predictor": predictor,
              "report": report}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="model-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dataset, predictor, report
