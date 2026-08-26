"""Final perceived-safety model and participant-independent evaluation.

The implementation is deliberately self-contained so the repository does not
depend on historical exploratory scripts. It reads the first five columns of
the supplied workbook, converts the source speed-scale variable to actual
robot motion speed (omega = 1.5 * v_s), fits a cumulative ordered-probit
model, and evaluates the nine-shot participant-intercept adaptation protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp, ndtr

N_LEVELS = 10
SPEED_SCALE_TO_RAD_S = 1.5
SEED = 20260823
SPEED_EDGES = np.array([0.3, 0.5, 0.75, 0.9])
DISTANCE_EDGES = np.array([0.1, 0.4, 0.7, 0.9])
SPEED_CENTRES = np.array([0.4, 0.625, 0.825])
DISTANCE_CENTRES = np.array([0.25, 0.55, 0.8])
SPEED_WIDTHS = np.array([0.2, 0.25, 0.15])
DISTANCE_WIDTHS = np.array([0.3, 0.3, 0.2])


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@dataclass
class CumulativeOrderedProbit:
    """Cumulative ordered-probit model with ordered thresholds."""

    l2: float = 0.001
    maxiter: int = 2000
    beta_: np.ndarray | None = None
    thresholds_: np.ndarray | None = None
    result_: object | None = None

    def _unpack(self, params: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        beta = params[:n_features]
        raw = params[n_features:]
        thresholds = np.empty(N_LEVELS - 1, dtype=float)
        thresholds[0] = raw[0]
        thresholds[1:] = raw[0] + np.cumsum(_softplus(raw[1:]))
        return beta, thresholds

    def _loss_gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        n, n_features = X.shape
        beta, thresholds = self._unpack(params, n_features)
        eta = X @ beta
        upper_exists = y < N_LEVELS - 1
        lower_exists = y > 0
        upper_cdf = np.ones(n)
        lower_cdf = np.zeros(n)
        upper_pdf = np.zeros(n)
        lower_pdf = np.zeros(n)
        if np.any(upper_exists):
            z = thresholds[y[upper_exists]] - eta[upper_exists]
            upper_cdf[upper_exists] = ndtr(z)
            upper_pdf[upper_exists] = _normal_pdf(z)
        if np.any(lower_exists):
            z = thresholds[y[lower_exists] - 1] - eta[lower_exists]
            lower_cdf[lower_exists] = ndtr(z)
            lower_pdf[lower_exists] = _normal_pdf(z)
        probability = np.maximum(upper_cdf - lower_cdf, 1e-14)
        loss = -float(np.mean(np.log(probability))) + 0.5 * self.l2 * float(beta @ beta)

        d_eta = (upper_pdf - lower_pdf) / probability / n
        grad_beta = X.T @ d_eta + self.l2 * beta
        grad_threshold = np.zeros(N_LEVELS - 1, dtype=float)
        if np.any(upper_exists):
            np.add.at(grad_threshold, y[upper_exists], -upper_pdf[upper_exists] / probability[upper_exists] / n)
        if np.any(lower_exists):
            np.add.at(grad_threshold, y[lower_exists] - 1, lower_pdf[lower_exists] / probability[lower_exists] / n)

        raw = params[n_features:]
        grad_raw = np.empty_like(raw)
        grad_raw[0] = np.sum(grad_threshold)
        grad_raw[1:] = np.cumsum(grad_threshold[::-1])[::-1][1:] * expit(raw[1:])
        return loss, np.concatenate([grad_beta, grad_raw])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CumulativeOrderedProbit":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must have the same number of rows")
        if np.any((y < 0) | (y >= N_LEVELS)):
            raise ValueError("y must be encoded as integers from 0 to 9")
        initial_beta = np.zeros(X.shape[1], dtype=float)
        counts = np.bincount(y, minlength=N_LEVELS).astype(float)
        cumulative = (np.cumsum(counts)[:-1] + 0.5) / (len(y) + 1.0)
        initial_thresholds = np.array([_normal_quantile(p) for p in cumulative])
        initial_raw = np.empty(N_LEVELS - 1, dtype=float)
        initial_raw[0] = initial_thresholds[0]
        gaps = np.maximum(np.diff(initial_thresholds), 0.05)
        initial_raw[1:] = np.log(np.expm1(gaps))
        initial = np.concatenate([initial_beta, initial_raw])
        result = minimize(
            lambda p: self._loss_gradient(p, X, y),
            initial,
            method="BFGS",
            jac=True,
            options={"maxiter": self.maxiter, "gtol": 1e-7},
        )
        self.result_ = result
        self.beta_, self.thresholds_ = self._unpack(result.x, X.shape[1])
        if not result.success:
            raise RuntimeError(f"Ordered-probit optimization failed: {result.message}")
        return self

    def predict_proba(self, X: np.ndarray, offset: float | np.ndarray = 0.0) -> np.ndarray:
        if self.beta_ is None or self.thresholds_ is None:
            raise RuntimeError("Fit the model before prediction")
        X = np.asarray(X, dtype=float)
        offsets = np.atleast_1d(np.asarray(offset, dtype=float))
        eta = X @ self.beta_
        cumulative = ndtr(self.thresholds_[None, None, :] - eta[None, :, None] - offsets[:, None, None])
        probabilities = np.empty((len(offsets), len(X), N_LEVELS), dtype=float)
        probabilities[:, :, 0] = cumulative[:, :, 0]
        probabilities[:, :, 1:-1] = np.diff(cumulative, axis=2)
        probabilities[:, :, -1] = 1.0 - cumulative[:, :, -1]
        probabilities = np.maximum(probabilities, 1e-14)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        return probabilities[0] if np.ndim(offset) == 0 else probabilities


def _normal_quantile(p: float) -> float:
    """Small inverse-normal approximation for stable initialization."""
    from scipy.special import ndtri

    return float(ndtri(p))


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            source = pd.read_excel(path, sheet_name="experiment_data")
        except ValueError:
            source = pd.read_excel(path)
    else:
        source = pd.read_csv(path)
    if source.shape[1] < 5:
        raise ValueError("Input data must contain at least five columns")
    participant_col, group_col, speed_col, distance_col, score_col = source.columns[:5]
    participant_text = source[participant_col].astype(str)
    participant = pd.to_numeric(participant_text.str.extract(r"^(\d+)")[0], errors="raise").astype(int)
    clean = pd.DataFrame(
        {
            "participant_id": participant,
            "participant_raw": participant_text,
            "experiment_group": pd.to_numeric(source[group_col], errors="raise").astype(int),
            "speed_scale": pd.to_numeric(source[speed_col], errors="raise").astype(float),
            "distance_m": pd.to_numeric(source[distance_col], errors="raise").astype(float),
            "observed_score": pd.to_numeric(source[score_col], errors="raise").astype(int),
        }
    )
    if clean.isna().any().any():
        raise ValueError("Required modelling columns contain missing values")
    if not clean.observed_score.between(1, N_LEVELS).all():
        raise ValueError("Perceived-safety scores must be integers from 1 to 10")
    clean["motion_speed_rad_s"] = SPEED_SCALE_TO_RAD_S * clean["speed_scale"]
    return clean


def make_features(frame: pd.DataFrame) -> np.ndarray:
    if "motion_speed_rad_s" in frame:
        omega = frame["motion_speed_rad_s"].to_numpy(float)
    else:
        omega = SPEED_SCALE_TO_RAD_S * frame["speed_scale"].to_numpy(float)
    distance = frame["distance_m"].to_numpy(float)
    return np.column_stack([omega, distance, distance**2, omega * distance])


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X, dtype=float).mean(axis=0)
    scale = np.asarray(X, dtype=float).std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


def participant_folds(participant_ids: np.ndarray, n_splits: int = 5, seed: int = SEED):
    participants = np.unique(np.asarray(participant_ids, dtype=int))
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    return [
        (np.flatnonzero(~np.isin(participant_ids, held_out)), np.flatnonzero(np.isin(participant_ids, held_out)))
        for held_out in np.array_split(participants, n_splits)
    ]


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    confusion = np.zeros((N_LEVELS, N_LEVELS), dtype=float)
    np.add.at(confusion, (y_true, y_pred), 1.0)
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / max(confusion.sum(), 1.0)
    idx = np.arange(N_LEVELS)
    weights = ((idx[:, None] - idx[None, :]) / (N_LEVELS - 1)) ** 2
    denominator = float(np.sum(weights * expected))
    return float(1.0 - np.sum(weights * confusion) / denominator) if denominator > 0 else float("nan")


def ranked_probability_score(y: np.ndarray, probabilities: np.ndarray) -> float:
    observed_cumulative = (y[:, None] <= np.arange(N_LEVELS - 1)[None, :]).astype(float)
    predicted_cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    return float(np.mean(np.sum((predicted_cumulative - observed_cumulative) ** 2, axis=1) / (N_LEVELS - 1)))


def evaluate_predictions(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    score_values = np.arange(1, N_LEVELS + 1, dtype=float)
    expected = probabilities @ score_values
    rounded = np.clip(np.floor(expected + 0.5), 1, N_LEVELS).astype(int)
    modal = np.argmax(probabilities, axis=1) + 1
    residual = expected - (y + 1)
    true_probability = probabilities[np.arange(len(y)), y]
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "within_1_point": float(np.mean(np.abs(residual) <= 1.0)),
        "within_2_points": float(np.mean(np.abs(residual) <= 2.0)),
        "exact_rounded": float(np.mean(rounded == y + 1)),
        "exact_modal": float(np.mean(modal == y + 1)),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y, rounded - 1),
        "ranked_probability_score": ranked_probability_score(y, probabilities),
        "negative_log_likelihood": float(-np.mean(np.log(np.maximum(true_probability, 1e-14)))),
        "mean_observed_score": float(np.mean(y + 1)),
        "mean_expected_score": float(np.mean(expected)),
    }


def assign_grid(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["speed_cell"] = pd.cut(result["speed_scale"], bins=SPEED_EDGES, labels=False, include_lowest=True, right=True)
    result["distance_cell"] = pd.cut(result["distance_m"], bins=DISTANCE_EDGES, labels=False, include_lowest=True, right=True)
    if result[["speed_cell", "distance_cell"]].isna().any().any():
        raise ValueError("Records fall outside the 3x3 calibration grid")
    result["speed_cell"] = result["speed_cell"].astype(int)
    result["distance_cell"] = result["distance_cell"].astype(int)
    return result


def select_one_per_cell(frame: pd.DataFrame) -> np.ndarray:
    selected: list[int] = []
    for speed_cell in range(3):
        for distance_cell in range(3):
            candidates = frame[(frame.speed_cell == speed_cell) & (frame.distance_cell == distance_cell)]
            if len(candidates) == 0:
                raise ValueError("Every participant must cover all nine speed-distance cells")
            z = np.column_stack(
                [
                    (candidates.speed_scale.to_numpy(float) - SPEED_CENTRES[speed_cell]) / SPEED_WIDTHS[speed_cell],
                    (candidates.distance_m.to_numpy(float) - DISTANCE_CENTRES[distance_cell]) / DISTANCE_WIDTHS[distance_cell],
                ]
            )
            chosen = candidates.index[np.argmin(np.sum(z**2, axis=1))]
            selected.append(int(np.flatnonzero(frame.index.to_numpy() == chosen)[0]))
    return np.asarray(selected, dtype=int)


def _score_probabilities(model: CumulativeOrderedProbit, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return model.predict_proba(X, offsets)


def _estimate_offset_scale(model: CumulativeOrderedProbit, X: np.ndarray, y: np.ndarray, participants: np.ndarray) -> float:
    offsets = []
    curvature_variances = []
    for participant in np.unique(participants):
        use = participants == participant

        def objective(offset: float) -> float:
            probability = _score_probabilities(model, X[use], np.array([offset]))[0]
            return float(-np.sum(np.log(probability[np.arange(np.sum(use)), y[use]])))

        fit = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
        estimate = float(fit.x)
        offsets.append(estimate)
        step = 0.02
        curvature = (objective(estimate + step) - 2 * objective(estimate) + objective(estimate - step)) / step**2
        curvature_variances.append(1.0 / curvature if curvature > 1e-8 else 0.0)
    offsets = np.asarray(offsets)
    corrected_variance = max(float(np.var(offsets, ddof=1)) - float(np.mean(curvature_variances)), 0.05**2)
    return float(np.sqrt(corrected_variance))


def posterior_predictive(
    model: CumulativeOrderedProbit,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_evaluation: np.ndarray,
    sigma: float,
    grid_size: int = 2001,
) -> tuple[np.ndarray, float, float]:
    grid_limit = max(1.0, 5.0 * sigma)
    grid = np.linspace(-grid_limit, grid_limit, grid_size)
    log_posterior = -0.5 * (grid / sigma) ** 2
    calibration_probability = _score_probabilities(model, X_calibration, grid)
    selected = calibration_probability[:, np.arange(len(y_calibration)), y_calibration]
    log_posterior += np.sum(np.log(np.maximum(selected, 1e-14)), axis=1)
    log_posterior -= logsumexp(log_posterior)
    weights = np.exp(log_posterior)
    evaluation_probability = _score_probabilities(model, X_evaluation, grid)
    prediction = np.einsum("g,gik->ik", weights, evaluation_probability)
    prediction = np.maximum(prediction, 1e-14)
    prediction /= prediction.sum(axis=1, keepdims=True)
    posterior_mean = float(np.sum(weights * grid))
    posterior_sd = float(np.sqrt(np.sum(weights * (grid - posterior_mean) ** 2)))
    return prediction, posterior_mean, posterior_sd


def cross_validate(data: pd.DataFrame, seed: int = SEED) -> dict:
    data = assign_grid(data.reset_index(names="source_row"))
    y = data.observed_score.to_numpy(int) - 1
    participants = data.participant_id.to_numpy(int)
    population_prob = np.zeros((len(data), N_LEVELS), dtype=float)
    generic_prob = np.zeros((len(data), N_LEVELS), dtype=float)
    personalized_prob = np.zeros((len(data), N_LEVELS), dtype=float)
    fold_details = []
    calibration_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(participant_folds(participants, 5, seed), start=1):
        train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
        x_train_raw, x_test_raw = make_features(train), make_features(test)
        mean, scale = fit_scaler(x_train_raw)
        x_train, x_test = (x_train_raw - mean) / scale, (x_test_raw - mean) / scale
        model = CumulativeOrderedProbit().fit(x_train, y[train_idx])
        population_prob[test_idx] = model.predict_proba(x_test)
        sigma = _estimate_offset_scale(model, x_train, y[train_idx], train.participant_id.to_numpy(int))
        test_positions = {int(row): pos for pos, row in enumerate(test.source_row)}

        for participant in sorted(test.participant_id.unique()):
            frame = test[test.participant_id == participant].copy()
            positions = np.asarray([test_positions[int(row)] for row in frame.source_row])
            x_part = x_test[positions]
            y_part = frame.observed_score.to_numpy(int) - 1
            calibration = select_one_per_cell(frame)
            evaluation = np.setdiff1d(np.arange(len(frame)), calibration)
            personal, posterior_mean, posterior_sd = posterior_predictive(
                model, x_part[calibration], y_part[calibration], x_part[evaluation], sigma
            )
            global_eval = frame.source_row.to_numpy(int)[evaluation]
            generic_prob[global_eval] = population_prob[test_idx[positions[evaluation]]]
            personalized_prob[global_eval] = personal
            for local in calibration:
                calibration_rows.append(frame.iloc[int(local)].to_dict())

        fold_details.append(
            {
                "outer_fold": fold_id,
                "train_participants": int(train.participant_id.nunique()),
                "train_records": int(len(train)),
                "test_participants": int(test.participant_id.nunique()),
                "test_records": int(len(test)),
                "offset_prior_sigma": sigma,
                "converged": bool(model.result_.success),
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
                "standardized_coefficients": model.beta_.tolist(),
            }
        )

    evaluation_mask = generic_prob.sum(axis=1) > 0
    return {
        "population_oof": evaluate_predictions(y, population_prob),
        "nine_shot_matched_population": evaluate_predictions(y[evaluation_mask], generic_prob[evaluation_mask]),
        "nine_shot_personalized": evaluate_predictions(y[evaluation_mask], personalized_prob[evaluation_mask]),
        "n_population_oof": int(len(data)),
        "n_nine_shot_evaluation": int(evaluation_mask.sum()),
        "fold_details": fold_details,
        "calibration_records": pd.DataFrame(calibration_rows),
        "data_with_source_row": data,
        "population_probabilities": population_prob,
        "generic_probabilities": generic_prob,
        "personalized_probabilities": personalized_prob,
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
