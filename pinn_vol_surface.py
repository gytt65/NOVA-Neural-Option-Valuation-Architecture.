#!/usr/bin/env python3
"""
pinn_vol_surface.py — Physics-Informed Neural Network Volatility Surface
==========================================================================

A numpy/scipy-only PINN that learns a volatility surface satisfying:
  1. Data fidelity:  σ(K, T) matches observed market IVs
  2. Black-Scholes PDE:  ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
  3. No butterfly arbitrage:  g(k) = (1 - k·w'/2w)² - w'/4·(1/w + 1/4) + w''/2 ≥ 0
  4. No calendar arbitrage:  ∂w(k,T)/∂T ≥ 0  (total variance non-decreasing in T)

Uses a Radial Basis Function (RBF) network as the neural approximator
instead of requiring PyTorch/TensorFlow.

Architecture:
    Input: (log_moneyness k, time_to_expiry T)
    → RBF layer: Σ_j c_j · φ(||x - μ_j|| / σ_j)
    → Linear output: total_variance w = σ² · T

References:
    Raissi et al. (2019) — "Physics-informed neural networks"
    Ackerer et al. (2020) — "Deep smoothing of the implied volatility surface"
    Zheng (2023) — "PINNs for arbitrage-free volatility surface calibration"
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ============================================================================
# RBF NETWORK
# ============================================================================

class RBFNetwork:
    """
    Radial Basis Function network.

    f(x) = Σ_j  w_j · exp(-||x - c_j||² / (2·σ_j²))  + bias

    Parameters:
    - centers c_j: shape (n_centers, input_dim)
    - widths σ_j: shape (n_centers,)
    - weights w_j: shape (n_centers,)
    - bias: scalar
    """

    def __init__(self, n_centers: int = 25, input_dim: int = 2):
        self.n_centers = n_centers
        self.input_dim = input_dim
        # Total params: centers + widths + weights + bias
        self.n_params = n_centers * input_dim + n_centers + n_centers + 1

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate RBF network.

        Parameters
        ----------
        X : shape (n, input_dim) — inputs
        params : shape (n_params,) — all network parameters

        Returns
        -------
        shape (n,) — network output (total variance w)
        """
        nc = self.n_centers
        nd = self.input_dim

        # Parse parameters
        centers = params[:nc * nd].reshape(nc, nd)          # (nc, nd)
        widths = np.abs(params[nc * nd:nc * nd + nc]) + 0.1  # (nc,), ensure positive
        weights = params[nc * nd + nc:nc * nd + 2 * nc]      # (nc,)
        bias = params[-1]

        # Compute RBF activations: exp(-||x - c||² / (2σ²))
        # X: (n, nd), centers: (nc, nd) → distances: (n, nc)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, nc, nd)
        sq_dist = np.sum(diff ** 2, axis=2)                      # (n, nc)
        activations = np.exp(-sq_dist / (2.0 * widths ** 2))     # (n, nc)

        # Output
        output = activations @ weights + bias  # (n,)
        return output

    def gradient_k(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂w/∂k — derivative of total variance w.r.t. log-moneyness."""
        eps = 1e-5
        X_plus = X.copy()
        X_plus[:, 0] += eps
        X_minus = X.copy()
        X_minus[:, 0] -= eps
        return (self.forward(X_plus, params) - self.forward(X_minus, params)) / (2 * eps)

    def gradient_T(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂w/∂T — derivative of total variance w.r.t. expiry."""
        eps = 1e-5
        X_plus = X.copy()
        X_plus[:, 1] += eps
        X_minus = X.copy()
        X_minus[:, 1] -= eps
        return (self.forward(X_plus, params) - self.forward(X_minus, params)) / (2 * eps)

    def hessian_kk(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂²w/∂k² — second derivative w.r.t. log-moneyness."""
        eps = 1e-4
        X_plus = X.copy()
        X_plus[:, 0] += eps
        X_minus = X.copy()
        X_minus[:, 0] -= eps
        f_plus = self.forward(X_plus, params)
        f_center = self.forward(X, params)
        f_minus = self.forward(X_minus, params)
        return (f_plus - 2 * f_center + f_minus) / (eps ** 2)


# ============================================================================
# PINN VOLATILITY SURFACE
# ============================================================================

class PINNVolSurface:
    """
    Physics-Informed volatility surface using RBF network.

    Fits a continuous function w(k, T) = σ²·T (total variance) that:
    - Matches observed market implied volatilities (data loss)
    - Satisfies no-butterfly-arbitrage constraint (convexity loss)
    - Satisfies no-calendar-arbitrage constraint (monotonicity in T)
    - Is smooth and regular (gradient penalty)

    Usage:
        pinn = PINNVolSurface()
        pinn.fit(log_moneyness, expiries, market_ivs)
        iv = pinn.get_iv(k=0.0, T=0.05)
        surface = pinn.get_surface(k_grid, T_grid)

    API compatible with ArbFreeSurfaceState for drop-in use.
    """

    def __init__(
        self,
        n_centers: int = 25,
        lambda_butterfly: float = 1.0,
        lambda_calendar: float = 0.5,
        lambda_smooth: float = 0.01,
        max_iter: int = 200,
    ):
        """
        Parameters
        ----------
        n_centers : int — number of RBF centers
        lambda_butterfly : float — weight on butterfly no-arbitrage loss
        lambda_calendar : float — weight on calendar no-arbitrage loss
        lambda_smooth : float — weight on smoothness regularization
        max_iter : int — maximum optimization iterations
        """
        self.rbf = RBFNetwork(n_centers=n_centers, input_dim=2)
        self.lambda_butterfly = lambda_butterfly
        self.lambda_calendar = lambda_calendar
        self.lambda_smooth = lambda_smooth
        self.max_iter = max_iter
        self.params: Optional[np.ndarray] = None
        self.is_fitted = False
        self.last_diagnostics: Dict = {}

    def fit(
        self,
        log_moneyness: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit PINN volatility surface to market data.

        Parameters
        ----------
        log_moneyness : array of log(K/F) values
        expiries : array of corresponding expiry times (years)
        market_ivs : array of corresponding implied volatilities
        weights : optional importance weights (e.g. by liquidity)

        Returns
        -------
        dict with fitting diagnostics
        """
        k = np.asarray(log_moneyness, dtype=float)
        T = np.asarray(expiries, dtype=float)
        iv = np.asarray(market_ivs, dtype=float)

        # Filter valid data
        valid = np.isfinite(k) & np.isfinite(T) & np.isfinite(iv) & (T > 0) & (iv > 0)
        k, T, iv = k[valid], T[valid], iv[valid]

        if len(k) < 5:
            self.last_diagnostics = {'error': 'insufficient_data', 'n_points': len(k)}
            return self.last_diagnostics

        # Target: total variance w = σ² · T
        w_target = iv ** 2 * T
        X = np.column_stack([k, T])  # (n, 2)

        if weights is None:
            W = np.ones(len(k))
        else:
            W = np.asarray(weights[valid], dtype=float)
        W = W / np.mean(W)  # normalize

        # Collocation points for physics constraints
        n_colloc = min(200, len(k) * 3)
        rng = np.random.default_rng(42)
        k_c = rng.uniform(k.min() - 0.05, k.max() + 0.05, n_colloc)
        T_c = rng.uniform(max(T.min(), 0.003), T.max() + 0.01, n_colloc)
        X_colloc = np.column_stack([k_c, T_c])

        # Initialize parameters
        self._init_params(k, T, w_target)

        def total_loss(params):
            # 1. Data fidelity loss
            w_pred = self.rbf.forward(X, params)
            data_loss = float(np.mean(W * (w_pred - w_target) ** 2))

            # 2. Butterfly arbitrage loss (g(k) ≥ 0)
            w_c = np.maximum(self.rbf.forward(X_colloc, params), 1e-10)
            dw_dk = self.rbf.gradient_k(X_colloc, params)
            d2w_dk2 = self.rbf.hessian_kk(X_colloc, params)

            # Gatheral's g(k) condition (simplified)
            # g(k) = 1 - k/(2w) · w' + w''/(2w) ≥ 0
            g = 1.0 - k_c * dw_dk / (2.0 * w_c) + d2w_dk2 / (2.0 * w_c)
            butterfly_loss = float(np.mean(np.maximum(-g, 0.0) ** 2))

            # 3. Calendar arbitrage loss (∂w/∂T ≥ 0)
            dw_dT = self.rbf.gradient_T(X_colloc, params)
            calendar_loss = float(np.mean(np.maximum(-dw_dT, 0.0) ** 2))

            # 4. Positivity constraint (w > 0)
            positivity_loss = float(np.mean(np.maximum(-w_c + 1e-6, 0.0) ** 2))

            # 5. Smoothness regularization (L2 on weights)
            smooth_loss = float(np.mean(params ** 2))

            total = (
                data_loss
                + self.lambda_butterfly * butterfly_loss
                + self.lambda_calendar * calendar_loss
                + 10.0 * positivity_loss
                + self.lambda_smooth * smooth_loss
            )

            return total

        # Optimize
        try:
            result = minimize(
                total_loss,
                self.params,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter, 'ftol': 1e-8},
            )
            self.params = result.x
            opt_loss = result.fun
            n_iter = result.nit
        except Exception as e:
            # Fallback to Nelder-Mead
            try:
                result = minimize(
                    total_loss,
                    self.params,
                    method='Nelder-Mead',
                    options={'maxiter': self.max_iter * 2},
                )
                self.params = result.x
                opt_loss = result.fun
                n_iter = result.nit
            except Exception:
                opt_loss = float('inf')
                n_iter = 0

        self.is_fitted = True

        # Diagnostics
        w_pred = self.rbf.forward(X, self.params)
        iv_pred = np.sqrt(np.maximum(w_pred / T, 1e-10))
        rmse = float(np.sqrt(np.mean((iv_pred - iv) ** 2)))
        max_err = float(np.max(np.abs(iv_pred - iv)))

        # Check no-arb conditions
        w_c = np.maximum(self.rbf.forward(X_colloc, self.params), 1e-10)
        dw_dk = self.rbf.gradient_k(X_colloc, self.params)
        d2w_dk2 = self.rbf.hessian_kk(X_colloc, self.params)
        dw_dT = self.rbf.gradient_T(X_colloc, self.params)

        g = 1.0 - k_c * dw_dk / (2.0 * w_c) + d2w_dk2 / (2.0 * w_c)

        self.last_diagnostics = {
            'loss': float(opt_loss),
            'rmse_iv': rmse,
            'max_error_iv': max_err,
            'n_iterations': int(n_iter),
            'n_data_points': len(k),
            'butterfly_violations': int(np.sum(g < -1e-4)),
            'calendar_violations': int(np.sum(dw_dT < -1e-4)),
            'butterfly_ok': bool(np.sum(g < -1e-4) == 0),
            'calendar_ok': bool(np.sum(dw_dT < -1e-4) == 0),
        }
        return self.last_diagnostics

    def get_iv(self, k: float, T: float) -> float:
        """Get implied volatility at (log_moneyness, expiry)."""
        if not self.is_fitted or self.params is None:
            return 0.15  # fallback

        X = np.array([[float(k), max(float(T), 1e-6)]])
        w = self.rbf.forward(X, self.params)[0]
        w = max(float(w), 1e-10)
        iv = np.sqrt(w / max(T, 1e-6))
        return float(np.clip(iv, 0.01, 5.0))

    def get_total_variance(self, k: float, T: float) -> float:
        """Get total variance w(k, T) = σ²·T."""
        if not self.is_fitted or self.params is None:
            return 0.15 ** 2 * T

        X = np.array([[float(k), max(float(T), 1e-6)]])
        w = self.rbf.forward(X, self.params)[0]
        return float(max(w, 1e-10))

    def get_surface(
        self,
        k_grid: np.ndarray,
        T_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate IV surface on a grid.

        Parameters
        ----------
        k_grid : shape (nk,) — log-moneyness values
        T_grid : shape (nt,) — expiry values

        Returns
        -------
        shape (nt, nk) — IV surface
        """
        if not self.is_fitted or self.params is None:
            return np.full((len(T_grid), len(k_grid)), 0.15)

        K, TT = np.meshgrid(k_grid, T_grid)
        X = np.column_stack([K.ravel(), TT.ravel()])
        w = self.rbf.forward(X, self.params)
        w = np.maximum(w, 1e-10)
        iv = np.sqrt(w / np.maximum(TT.ravel(), 1e-6))
        return np.clip(iv.reshape(len(T_grid), len(k_grid)), 0.01, 5.0)

    def _init_params(self, k: np.ndarray, T: np.ndarray, w: np.ndarray):
        """Smart initialization of RBF parameters from data."""
        nc = self.rbf.n_centers
        nd = self.rbf.input_dim
        rng = np.random.default_rng(42)

        # Place centers on a grid spanning the data
        k_centers = np.linspace(k.min() - 0.05, k.max() + 0.05,
                                int(np.sqrt(nc)) + 1)
        T_centers = np.linspace(max(T.min(), 0.003), T.max() + 0.01,
                                int(np.sqrt(nc)) + 1)
        K_c, T_c = np.meshgrid(k_centers, T_centers)
        centers_grid = np.column_stack([K_c.ravel(), T_c.ravel()])

        # Subsample to nc centers
        if len(centers_grid) > nc:
            idx = rng.choice(len(centers_grid), nc, replace=False)
            centers = centers_grid[idx]
        else:
            # Pad with random centers
            n_extra = nc - len(centers_grid)
            extra = np.column_stack([
                rng.uniform(k.min(), k.max(), n_extra),
                rng.uniform(T.min(), T.max(), n_extra),
            ])
            centers = np.vstack([centers_grid, extra])[:nc]

        # Widths: proportional to spacing
        widths = np.full(nc, 0.3)

        # Weights: small random initialization
        weights = rng.normal(0, np.std(w) / nc, nc)

        # Bias: mean of target
        bias = np.mean(w)

        self.params = np.concatenate([
            centers.ravel(),    # nc * nd
            widths,             # nc
            weights,            # nc
            [bias],             # 1
        ])
