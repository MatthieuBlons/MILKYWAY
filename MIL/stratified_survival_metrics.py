"""
stratified_survival_metrics.py
================================

Correct evaluation of **stratified Cox proportional-hazards models**
(``lifelines.CoxPHFitter(strata=...)``) using:

1. **Harrell's concordance index (C-index)**, computed *within* each
   stratum and pooled by summing concordant / discordant / tied pair
   counts across strata.
2. **Cumulative/dynamic AUC(t)** (the Hung & Chiang / Uno-style IPCW
   estimator implemented in ``sksurv.metrics.cumulative_dynamic_auc``),
   using the model-implied absolute risk ``1 - S(t | x)`` as the
   time-varying risk score, instead of the (stratum-agnostic) linear
   predictor.

Why these need special handling for a *stratified* Cox model
--------------------------------------------------------------
A stratified Cox model assumes a *shared* coefficient vector ``beta``
but a *separate* baseline hazard ``h0_s(t)`` per stratum ``s``. That
has two consequences for evaluation:

* **The linear predictor is only meaningful within a stratum.**
  ``LP_i = x_i . beta`` tells you how subject ``i``'s hazard compares
  to other subjects *in the same stratum* (since ``h0_s(t)`` cancels
  out of within-stratum contrasts). It says nothing about how baseline
  risk compares *across* strata, because two subjects in different
  strata can have identical ``LP`` and still have very different
  absolute risk if ``h0_s(t)`` differs. So Harrell's C-index (which
  compares subjects pairwise by risk score) must restrict comparisons
  to pairs within the same stratum -- exactly what
  :func:`stratified_concordance_index` below does.

* **The absolute risk ``1 - S(t | x)`` does not have this problem.**
  ``S(t | x) = S0_s(t) ** exp(LP_i)`` is an actual (stratum-specific)
  survival probability, so ``1 - S(t | x)`` is directly comparable
  *across* strata: it always means "probability this subject has
  failed by time t," regardless of which stratum they belong to. This
  is precisely why substituting ``1 - S(t)`` for the linear predictor
  is the right fix for computing a *pooled, cross-stratum*
  cumulative/dynamic AUC(t) -- unlike the C-index, it does **not**
  need to be restricted to within-stratum comparisons.

A subtle correctness pitfall this module guards against
----------------------------------------------------------
``lifelines.CoxPHFitter.predict_survival_function`` internally groups
rows by stratum before computing survival curves, and returns columns
in that *stratum-grouped* order rather than the caller's original row
order. Naively assigning the resulting risk scores back to rows (e.g.
via ``.values``) silently misaligns subjects and risk scores whenever
more than one stratum is present. Every function below explicitly
reindexes on the input DataFrame's index to guard against this.

Requirements: ``lifelines``, ``scikit-survival`` (``sksurv``), ``numpy``,
``pandas``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

__all__ = [
    "to_structured_survival_array",
    "predicted_risk_at_times",
    "stratified_concordance_index",
    "cumulative_dynamic_auc_from_survival",
    "evaluate_stratified_cox_model",
]


# ── data plumbing ────────────────────────────────────────────────────────────

def to_structured_survival_array(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """
    Build the structured array ``sksurv`` expects: a record array with a
    boolean ``event`` field and a float ``time`` field.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        Observed time (event or censoring time).
    event : array-like of shape (n_samples,)
        1/True if the event was observed, 0/False if censored.

    Returns
    -------
    np.ndarray
        Structured array with dtype ``[('event', bool), ('time', float)]``.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    y = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    y["event"] = event
    y["time"] = time
    return y


# ── absolute risk 1 - S(t | x), correctly respecting strata for Lifelines ────────────────

def predicted_risk_at_times(
    cph,
    df: pd.DataFrame,
    times: Sequence[float],
) -> pd.DataFrame:
    """
    Compute the model-implied absolute risk ``1 - S(t | x)`` for every row
    of ``df`` at every requested time, for a (possibly stratified)
    ``lifelines.CoxPHFitter``.

    This delegates the actual survival-curve computation to
    ``cph.predict_survival_function``, which already resolves each row to
    the correct per-stratum baseline hazard -- so there is no need to
    manually look up ``cph.baseline_cumulative_hazard_`` columns by
    stratum label (which is brittle: those columns can be stringified,
    reordered, or use tuples for multiple strata columns).

    Importantly, this function reindexes the result back onto ``df``'s
    original row order. ``predict_survival_function`` internally groups
    rows by stratum, so its output columns come back in a
    *stratum-grouped* order rather than ``df``'s order -- silently
    returning them unindexed would misalign risk scores with subjects
    whenever ``cph`` was fit with more than one stratum.

    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        A fitted (optionally stratified) Cox model.
    df : pd.DataFrame
        Covariate rows to score. Must have a unique index. If ``cph`` was
        fit with ``strata=[...]``, those columns must be present in ``df``.
    times : sequence of float
        Times at which to evaluate risk. ``S(t)`` is a right-continuous
        step function, so times that fall between observed event times
        use the most recently observed step, and times beyond the last
        observed time use the final (possibly non-zero) survival estimate
        -- i.e. no extrapolation beyond the data.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, n_times), indexed exactly like ``df.index``,
        columns equal to ``times``. Entry (i, t) = ``1 - S(t | x_i)``.
    """
    if not df.index.is_unique:
        raise ValueError(
            "df must have a unique index so risk scores can be safely "
            "reindexed back onto the input row order."
        )

    times = list(times)
    sf = cph.predict_survival_function(df, times=times)  # index=times, columns=df.index (reordered!)
    risk = (1.0 - sf).T                                   # rows back to per-subject
    risk = risk.reindex(df.index)                          # undo lifelines' stratum-grouped reordering
    risk.columns = times

    if risk.isna().any().any():
        raise ValueError(
            "NaNs in predicted risk -- check that every row's strata "
            "value(s) were seen during cph.fit()."
        )
    return risk


# ── Harrell's C-index, pooled across strata ─────────────────────────────────

def stratified_concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    strata: np.ndarray,
) -> Tuple[float, dict]:
    """
    Harrell's concordance index, computed within each stratum and pooled
    by summing comparable-pair counts across strata.

    Comparisons are restricted to pairs within the same stratum because
    ``risk_score`` is expected to be something like the Cox linear
    predictor, which (per the module docstring) is only meaningful as a
    *relative* ranking within a stratum -- comparing it across strata
    would score pairs as concordant/discordant based on differences in
    baseline hazard that the score was never meant to capture.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        Observed time (event or censoring time).
    event : array-like of shape (n_samples,)
        1/True if the event was observed, 0/False if censored.
    risk_score : array-like of shape (n_samples,)
        Higher = higher risk (e.g. the Cox linear predictor, or
        ``1 - S(t0)`` at some fixed horizon ``t0``).
    strata : array-like of shape (n_samples,)
        Stratum label per row. Only pairs sharing the same label are
        compared.

    Returns
    -------
    cindex : float
        Pooled Harrell's C-index across all strata.
    counts : dict
        Pooled concordant / discordant / tied_risk / tied_time /
        tied_both pair counts, for auditing or computing your own
        variant of the index.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    risk_score = np.asarray(risk_score, dtype=float)
    strata = np.asarray(strata)

    if not (len(time) == len(event) == len(risk_score) == len(strata)):
        raise ValueError("time, event, risk_score, and strata must be the same length.")

    counts = {"concordant": 0.0, "discordant": 0.0, "tied_risk": 0.0, "tied_time": 0.0, "tied_both": 0.0}

    for s in np.unique(strata):
        idx = strata == s
        n_events = event[idx].sum()
        if idx.sum() < 2 or n_events == 0:
            # Not enough data / no events in this stratum to form any
            # comparable pair -- skip rather than let sksurv raise.
            continue

        try:
            cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
                event[idx], time[idx], risk_score[idx]
            )[:5]
        except NoComparablePairException:
            # Events and censorings in this stratum form no comparable
            # pair (e.g. the only event comes after every censoring time).
            continue

        counts["concordant"] += concordant
        counts["discordant"] += discordant
        counts["tied_risk"] += tied_risk
        counts["tied_time"] += tied_time

    comparable = counts["concordant"] + counts["discordant"] + counts["tied_risk"]
    if comparable == 0:
        raise ValueError("No comparable pairs found in any stratum.")

    pooled_cindex = (counts["concordant"] + 0.5 * counts["tied_risk"]) / comparable
    return pooled_cindex, counts


# ── cumulative/dynamic AUC(t), using 1 - S(t) so strata pool correctly ─────

def cumulative_dynamic_auc_from_survival(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_at_times: pd.DataFrame,
    times: Sequence[float],
) -> Tuple[np.ndarray, float]:
    """
    Wrapper around ``sksurv.metrics.cumulative_dynamic_auc``
    that takes a precomputed *time-varying* risk score, i.e. shape
    (n_samples, n_times) rather than a single time-invariant score.

    Because ``risk_at_times`` should be built from ``1 - S(t | x)`` (see
    :func:`predicted_risk_at_times`), it is on an absolute-probability
    scale that is meaningful across strata.

    Parameters
    ----------
    y_train : structured array
        Survival outcomes (from :func:`to_structured_survival_array`) used
        to estimate the censoring distribution for IPCW weighting. Pass
        the *training* set's outcomes if ``y_test`` is a held-out set;
        for in-sample evaluation, pass the same data as ``y_test``.
    y_test : structured array
        Survival outcomes for the subjects being scored. Row order must
        match ``risk_at_times``.
    risk_at_times : pd.DataFrame
        Output of :func:`predicted_risk_at_times`: shape
        (n_samples, n_times), row order matching ``y_test``.
    times : sequence of float
        Times at which AUC(t) is evaluated. Must match the columns of
        ``risk_at_times``, and (per sksurv's requirement) must lie
        strictly within the observed follow-up range of ``y_train``.

    Returns
    -------
    auc_at_times : np.ndarray of shape (n_times,)
        AUC(t) for each requested time.
    mean_auc : float
        AUC(t) averaged over ``times``, weighted as sksurv does (by the
        estimated fraction of events observed up to each time).
    """
    times = list(times)
    if list(risk_at_times.columns) != times:
        raise ValueError("risk_at_times.columns must match `times`, in the same order.")
    if len(risk_at_times) != len(y_test):
        raise ValueError("risk_at_times and y_test must have the same number of rows.")

    auc_at_times, mean_auc = cumulative_dynamic_auc(
        y_train, y_test, risk_at_times.to_numpy(), times
    )
    return auc_at_times, mean_auc


