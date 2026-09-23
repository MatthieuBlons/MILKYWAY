"""
Custom losses for MIL.

Supported tasks
---------------
survival
    Predict a continuous risk score from observed survival times and
    censoring indicators.

classification
    Predict a binary or multiclass categorical endpoint.

regression
    Predict one or multiple continuous targets.
"""

from __future__ import annotations


import torch
from torch import Tensor
from torch.nn import (
    Module,
)


class CoxPartialLikelihoodLoss(Module):
    """CoxPartialLikelihoodLoss.

    Negative Cox proportional-hazards partial log-likelihood.

    This implementation uses the Breslow approximation for tied event times.

    Notes
    -----
    The Cox loss depends on the composition of the risk set. Computing it
    independently on small minibatches gives only a minibatch approximation
    of the full-cohort partial likelihood.

    Parameters
    ----------
    reduction : {"mean", "sum"}, default="mean"
        Reduction applied over the events of the batch.
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        risk_scores: Tensor,
        times: Tensor,
        events: Tensor,
    ) -> Tensor:
        """
        Compute the negative partial log-likelihood.

        Parameters
        ----------
        risk_scores : torch.Tensor
            Predicted log-risk scores with shape (B,) or (B, 1).

        times : torch.Tensor
            Observed survival or follow-up times with shape (B,).

        events : torch.Tensor
            Event indicators with shape (B,).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        risk_scores = risk_scores.reshape(-1)
        times = times.reshape(-1)
        events = events.reshape(-1).bool()

        if not events.any():
            # Maintain a differentiable zero when a batch contains no events.
            return risk_scores.sum() * 0.0

        order = torch.argsort(
            times,
            descending=True,
        )

        ordered_risk = risk_scores[order]
        ordered_events = events[order]

        log_cumulative_risk = torch.logcumsumexp(
            ordered_risk,
            dim=0,
        )

        loss = (ordered_risk - log_cumulative_risk)[ordered_events]

        # Apply reduction
        if self.reduction == "mean":
            return -loss.mean()
        elif self.reduction == "sum":
            return -loss.sum()
        else:
            raise ValueError(
                f"Invalid reduction type: {self.reduction}. Must be 'mean' or 'sum'."
            )


class NLLSurvLoss(Module):
    """
    Discrete-time negative log-likelihood loss for survival analysis.

    Parameters
    ----------
    alpha : float, optional
        Optional weighting of uncensored observations. Default is 0.0.

    reduction : {"mean", "sum"}, optional
        Reduction applied over the batch. Default is "mean".

    eps : float, optional
        Numerical stability constant. Default is 1e-7.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()

        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        time_bin: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the discrete-time survival negative log-likelihood.

        Parameters
        ----------
        logits : torch.Tensor
            Network outputs with shape (B, n_bins), where n_bins is the number
            of discrete survival-time bins.
        time_bin : torch.Tensor
            Discrete survival-time bin indices with shape [B].
            Values must be in [0, n_bins - 1].
        event : torch.Tensor
            Event indicator with shape (B,).
            1 indicates an observed event and 0 indicates censoring.

        Returns
        -------
        torch.Tensor
            Scalar mean survival loss.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B, n_bins), got {logits.shape}.")

        time_bin = time_bin.long().view(-1)
        event = event.float().view(-1)

        hazards = torch.sigmoid(logits)

        survival = torch.cumprod(
            1.0 - hazards,
            dim=1,
        )

        survival = torch.cat(
            [
                torch.ones(
                    (logits.shape[0], 1),
                    dtype=survival.dtype,
                    device=survival.device,
                ),
                survival,
            ],
            dim=1,
        )

        time_bin = time_bin.unsqueeze(1)
        event = event.unsqueeze(1)

        survival_before = torch.gather(
            survival,
            dim=1,
            index=time_bin,
        ).clamp_min(self.eps)

        survival_through = torch.gather(
            survival,
            dim=1,
            index=time_bin + 1,
        ).clamp_min(self.eps)

        hazard_at_event = torch.gather(
            hazards,
            dim=1,
            index=time_bin,
        ).clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        uncensored_loss = -event * (
            torch.log(survival_before) + torch.log(hazard_at_event)
        )

        censored_loss = -(1.0 - event) * torch.log(survival_through)

        loss = uncensored_loss + (1 - self.alpha) * censored_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid reduction type: {self.reduction}. Must be 'mean' or 'sum'."
            )
