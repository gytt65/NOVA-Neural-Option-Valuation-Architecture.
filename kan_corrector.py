#!/usr/bin/env python3
"""
kan_corrector.py — Kolmogorov-Arnold Network Pricing Corrector
===============================================================

A numpy-only implementation of KAN (Kolmogorov-Arnold Networks) for
option pricing correction. Drop-in replacement for the GradientBoosting-based
MLPricingCorrector with the same API surface.

Key differences from MLP/GBM:
    - Learnable activation functions on EDGES (B-splines), not fixed activations on nodes
    - More interpretable: can extract symbolic pricing rules from learned splines
    - Better extrapolation: splines generalize more naturally than piecewise-constant trees
    - Smaller networks needed: [n_features, 8, 4, 1] suffices vs 200-tree ensembles

Architecture:
    Input → [B-spline edges] → Hidden_1(8) → [B-spline edges] → Hidden_2(4) → [B-spline edges] → Output(1)

Each edge has its own learnable B-spline with k knots, parameterized by
control point coefficients optimized via L-BFGS-B.

References:
    Liu et al. (2024) — "KAN: Kolmogorov-Arnold Networks"
    Bozorgasl & Chen (2024) — "Wav-KAN: Wavelet Kolmogorov-Arnold Networks"
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ============================================================================
# B-SPLINE BASIS
# ============================================================================

def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate B-spline basis functions at points x.

    Parameters
    ----------
    x : shape (n,) — evaluation points (should be in [knots[degree], knots[-degree-1]])
    knots : shape (n_knots,) — augmented knot vector
    degree : int — spline degree (3 = cubic)

    Returns
    -------
    shape (n, n_basis) — basis function values, where n_basis = len(knots) - degree - 1
    """
    n_basis = len(knots) - degree - 1
    n_pts = len(x)
    B = np.zeros((n_pts, n_basis))

    # Degree 0: indicator functions
    for i in range(n_basis + degree):
        if i < len(knots) - 1:
            mask = (x >= knots[i]) & (x < knots[i + 1])
            if i < n_basis:
                B[mask, i] = 1.0

    # Cox-de Boor recursion
    for d in range(1, degree + 1):
        B_new = np.zeros((n_pts, n_basis))
        for i in range(n_basis + degree - d):
            if i < n_basis:
                # Left term
                denom1 = knots[i + d] - knots[i]
                if denom1 > 1e-10:
                    if i < B.shape[1]:
                        B_new[:, i] += B[:, i] * (x - knots[i]) / denom1

                # Right term
                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 > 1e-10 and (i + 1) < B.shape[1]:
                    B_new[:, i] += B[:, i + 1] * (knots[i + d + 1] - x) / denom2

        B = B_new

    # Handle right boundary
    B[-1 == np.argmax(x), -1] = 1.0 if n_basis > 0 else 0.0

    return B


def _make_knots(n_internal: int, degree: int = 3,
                x_min: float = -3.0, x_max: float = 3.0) -> np.ndarray:
    """Create augmented knot vector for B-spline."""
    internal = np.linspace(x_min, x_max, n_internal + 2)
    knots = np.concatenate([
        np.full(degree, x_min),
        internal,
        np.full(degree, x_max),
    ])
    return knots


# ============================================================================
# KAN LAYER
# ============================================================================

class KANLayer:
    """
    One layer of Kolmogorov-Arnold Network.

    Each edge (i, j) from input_dim to output_dim has its own B-spline
    parameterized by `n_spline_params` control points.

    Forward pass:
        output_j = Σ_i  spline_{i,j}(input_i)  +  bias_j
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_spline_knots: int = 5, spline_degree: int = 3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree

        # Knots (shared across all edges in the layer)
        self.knots = _make_knots(n_spline_knots, spline_degree)
        self.n_basis = len(self.knots) - spline_degree - 1

        # Parameters: control points for each edge spline + biases
        # Shape: (input_dim, output_dim, n_basis) for spline coefficients
        # Shape: (output_dim,) for biases
        self.n_params = input_dim * output_dim * self.n_basis + output_dim

    def forward(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Forward pass through this KAN layer.

        Parameters
        ----------
        x : shape (batch, input_dim) — input activations
        params : shape (n_params,) — flattened parameters for this layer

        Returns
        -------
        shape (batch, output_dim) — output activations
        """
        batch = x.shape[0]

        # Extract spline coefficients and biases
        n_coeff = self.input_dim * self.output_dim * self.n_basis
        coeffs = params[:n_coeff].reshape(self.input_dim, self.output_dim, self.n_basis)
        biases = params[n_coeff:]

        output = np.zeros((batch, self.output_dim))

        for i in range(self.input_dim):
            # Clip inputs to knot range for numerical stability
            x_i = np.clip(x[:, i], self.knots[self.spline_degree],
                          self.knots[-self.spline_degree - 1] - 1e-6)
            # Evaluate B-spline basis
            B = _bspline_basis(x_i, self.knots, self.spline_degree)  # (batch, n_basis)

            for j in range(self.output_dim):
                # Apply spline: sum of basis * coefficients
                output[:, j] += B @ coeffs[i, j, :]  # (batch,)

        output += biases  # broadcast (output_dim,)
        return output


# ============================================================================
# KAN NETWORK
# ============================================================================

class KANCorrector:
    """
    Kolmogorov-Arnold Network for option pricing correction.

    Architecture: Input(n_features) → KAN_Layer(8) → KAN_Layer(4) → KAN_Layer(1)

    API compatible with MLPricingCorrector:
        - predict_correction(features) → (correction, confidence)
        - add_sample(features, residual)
        - get_feature_importance()

    Advantages over GBM:
        1. Interpretable: learned splines can be visualized per feature
        2. Better extrapolation: splines vs piecewise-constant
        3. Compact: ~500 params vs 200 trees × many leaves

    Usage:
        kan = KANCorrector()
        kan.add_sample(features, residual)
        # ... add more samples ...
        correction, confidence = kan.predict_correction(features)
    """

    MIN_SAMPLES = 30
    RETRAIN_EVERY = 20

    def __init__(self, hidden_dims: Tuple[int, ...] = (8, 4),
                 n_spline_knots: int = 5, spline_degree: int = 3,
                 feature_names: Optional[List[str]] = None):
        self.hidden_dims = hidden_dims
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree
        self.feature_names = feature_names

        self.layers: List[KANLayer] = []
        self.params: Optional[np.ndarray] = None
        self.is_trained = False
        self.training_X: List[list] = []
        self.training_y: List[float] = []
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def _build_network(self, n_features: int):
        """Construct KAN layers."""
        self._n_features = n_features
        self.layers = []
        dims = [n_features] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            layer = KANLayer(dims[i], dims[i + 1],
                             self.n_spline_knots, self.spline_degree)
            self.layers.append(layer)

        # Total parameter count
        total_params = sum(layer.n_params for layer in self.layers)
        return total_params

    def _forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Full forward pass through the KAN.

        Parameters
        ----------
        X : shape (batch, n_features)
        params : shape (total_params,) — all network parameters

        Returns
        -------
        shape (batch, 1) — predictions
        """
        # Standardize inputs
        if self._input_mean is not None:
            X = (X - self._input_mean) / np.maximum(self._input_std, 1e-8)

        h = X
        offset = 0
        for layer in self.layers:
            p = params[offset:offset + layer.n_params]
            h = layer.forward(h, p)
            # Activation between hidden layers (not on output)
            if layer != self.layers[-1]:
                h = np.tanh(h)  # secondary activation for numerical stability
            offset += layer.n_params

        return h

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray,
              l2_reg: float = 0.01) -> float:
        """MSE loss with L2 regularization on spline coefficients."""
        pred = self._forward(X, params).ravel()
        mse = np.mean((pred - y) ** 2)
        reg = l2_reg * np.mean(params ** 2)
        return float(mse + reg)

    def _loss_grad(self, params: np.ndarray, X: np.ndarray, y: np.ndarray,
                   l2_reg: float = 0.01) -> np.ndarray:
        """Numerical gradient of loss (finite differences)."""
        grad = np.zeros_like(params)
        eps = 1e-5
        loss0 = self._loss(params, X, y, l2_reg)

        # Use forward differences for speed
        for i in range(len(params)):
            params[i] += eps
            loss1 = self._loss(params, X, y, l2_reg)
            grad[i] = (loss1 - loss0) / eps
            params[i] -= eps

        return grad

    def _train(self):
        """Train KAN via L-BFGS-B optimization."""
        if len(self.training_X) < self.MIN_SAMPLES:
            return

        X = np.array(self.training_X, dtype=float)
        y = np.array(self.training_y, dtype=float)
        n_features = X.shape[1]

        # Standardize
        self._input_mean = np.mean(X, axis=0)
        self._input_std = np.std(X, axis=0) + 1e-8
        X_std = (X - self._input_mean) / self._input_std

        # Build network if needed
        if not self.layers or self._n_features != n_features:
            total_params = self._build_network(n_features)
        else:
            total_params = sum(layer.n_params for layer in self.layers)

        # Initialize parameters (small random)
        rng = np.random.default_rng(42)
        if self.params is None or len(self.params) != total_params:
            self.params = rng.normal(0, 0.1, total_params)

        # Optimize with L-BFGS-B (don't use gradient — too expensive for numerical)
        # Use Nelder-Mead for small param counts, L-BFGS-B for larger
        try:
            result = minimize(
                self._loss,
                self.params,
                args=(X, y, 0.01),
                method='L-BFGS-B',
                jac=lambda p: self._loss_grad(p, X, y, 0.01),
                options={'maxiter': 50, 'ftol': 1e-6},
            )
            self.params = result.x
        except Exception:
            # Fallback to Nelder-Mead (gradient-free)
            try:
                result = minimize(
                    self._loss,
                    self.params,
                    args=(X, y, 0.01),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5},
                )
                self.params = result.x
            except Exception:
                pass

        self.is_trained = True

    # ------------------------------------------------------------------
    # PUBLIC API (same as MLPricingCorrector)
    # ------------------------------------------------------------------

    def predict_correction(self, features: dict) -> Tuple[float, float]:
        """
        Returns (correction_factor, confidence).

        correction is a fraction (e.g. 0.03 ⇒ +3% adjustment to NIRV price).
        """
        if not self.is_trained or self.params is None:
            return 0.0, 0.0

        try:
            X = self._features_to_array(features).reshape(1, -1)
            pred = self._forward(X, self.params).ravel()[0]
            correction = float(np.clip(pred, -0.20, 0.20))
            confidence = min(0.85, len(self.training_X) / 500.0)
            return correction, confidence
        except Exception:
            return 0.0, 0.0

    def add_sample(self, features: dict, residual: float):
        """Add training sample; auto-retrain with adaptive interval."""
        X = self._features_to_array(features)
        self.training_X.append(X.tolist())
        self.training_y.append(float(residual))

        n = len(self.training_X)
        retrain_interval = max(self.RETRAIN_EVERY, n // 15)
        if n >= self.MIN_SAMPLES and n % retrain_interval == 0:
            self._train()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        KAN interpretability: compute feature importance by ablation.

        For each feature, zero it out and measure increase in loss.
        Higher increase → more important feature.
        """
        if not self.is_trained or self.params is None or len(self.training_X) < 10:
            return {}

        X = np.array(self.training_X[-200:], dtype=float)
        y = np.array(self.training_y[-200:], dtype=float)
        base_loss = self._loss(self.params, X, y, 0.0)

        importance = {}
        names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]

        for i in range(min(X.shape[1], len(names))):
            X_ablated = X.copy()
            X_ablated[:, i] = 0.0  # zero out feature
            ablated_loss = self._loss(self.params, X_ablated, y, 0.0)
            imp = max(0.0, ablated_loss - base_loss) / max(base_loss, 1e-8)
            if imp > 0.005:
                importance[names[i]] = round(float(imp), 4)

        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def get_edge_splines(self, layer_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract learned spline functions for visualization/interpretation.

        Returns dict mapping (input_name, output_idx) to arrays of
        (x_values, y_values) for plotting.
        """
        if not self.is_trained or self.params is None or layer_idx >= len(self.layers):
            return {}

        layer = self.layers[layer_idx]
        offset = sum(self.layers[i].n_params for i in range(layer_idx))
        n_coeff = layer.input_dim * layer.output_dim * layer.n_basis
        coeffs = self.params[offset:offset + n_coeff].reshape(
            layer.input_dim, layer.output_dim, layer.n_basis
        )

        x_eval = np.linspace(-3, 3, 100)
        B = _bspline_basis(x_eval, layer.knots, layer.spline_degree)

        splines = {}
        names = self.feature_names or [f'f{i}' for i in range(layer.input_dim)]
        for i in range(layer.input_dim):
            for j in range(layer.output_dim):
                y_eval = B @ coeffs[i, j, :]
                key = f'{names[i] if i < len(names) else f"f{i}"} → h{j}'
                splines[key] = np.column_stack([x_eval, y_eval])

        return splines

    # ------------------------------------------------------------------
    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array."""
        if self.feature_names:
            return np.array([features.get(n, 0.0) for n in self.feature_names],
                            dtype=np.float64)
        # If no feature names set, use all dict values in sorted key order
        return np.array([features[k] for k in sorted(features.keys())],
                        dtype=np.float64)
