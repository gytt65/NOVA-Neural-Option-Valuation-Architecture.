#!/usr/bin/env python3
"""
neural_jsde.py — Neural Jump Stochastic Differential Equation Pricer
=====================================================================

The most advanced pricing model: drift, diffusion, AND jump parameters
are learned as functions of market state via neural networks.

Architecture:
    Traditional:  dS = μ·S·dt + σ·S·dW + J·dN(λ)     [fixed μ, σ, λ, J]
    Neural J-SDE: dS = μ_θ(X)·S·dt + σ_θ(X)·S·dW + J_θ(X)·dN(λ_θ(X))

    Where X = (S, t, VIX, regime_prob, FII_flow, PCR, ...) are market features
    and θ are neural network weights learned from historical option chains.

Training objective:
    Minimize Σ_i |Model_Price(K_i, T_i; θ) - Market_Price(K_i, T_i)|²
    across all strikes K and expiries T simultaneously.

This numpy-only implementation uses a 3-layer RBF network as the
"neural" component, trained via L-BFGS-B.

Key insight: The jump parameters (λ, μ_j, σ_j) become OUTPUT HEADS
of the neural network, allowing the model to learn state-dependent
dynamics that change with VIX, regime, and flows.

References:
    "Neural Jump Stochastic Differential Equation Model" — arXiv 2025
    "Neural SDEs for Option Pricing" — Gierjatowicz et al. 2022
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


class NeuralParameterNetwork:
    """
    A small RBF network that maps market state features to SDE parameters.

    Input:  X = [VIX/100, regime_crisis, regime_normal, regime_trending, log_moneyness, T]
    Output: [drift_adj, vol_mult, jump_intensity, jump_mean, jump_std]

    Compact architecture: 6 → 15 RBF centers → 5 outputs
    Total params: 15*6 (centers) + 15 (widths) + 15*5 (weights) + 5 (biases) = 185
    """

    def __init__(self, n_features: int = 6, n_centers: int = 15, n_outputs: int = 5):
        self.n_features = n_features
        self.n_centers = n_centers
        self.n_outputs = n_outputs
        self.n_params = (
            n_centers * n_features   # centers
            + n_centers              # widths
            + n_centers * n_outputs  # weights
            + n_outputs              # biases
        )

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate network.

        Parameters
        ----------
        X : shape (n, n_features)
        params : shape (n_params,)

        Returns
        -------
        shape (n, n_outputs) — raw SDE parameter outputs
        """
        nc, nf, no = self.n_centers, self.n_features, self.n_outputs

        # Parse params
        idx = 0
        centers = params[idx:idx + nc * nf].reshape(nc, nf)
        idx += nc * nf
        widths = np.abs(params[idx:idx + nc]) + 0.1
        idx += nc
        weights = params[idx:idx + nc * no].reshape(nc, no)
        idx += nc * no
        biases = params[idx:idx + no]

        # RBF activations
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        phi = np.exp(-sq_dist / (2.0 * widths ** 2))

        # Output = weighted sum + bias
        return phi @ weights + biases


class NeuralJSDE:
    """
    Neural Jump-SDE pricer.

    The SDE dynamics are parameterized by a neural network:
        μ(X), σ(X), λ(X), μ_j(X), σ_j(X) = NetworkOutput(features)

    where features include VIX, regime probabilities, moneyness, and expiry.

    Usage:
        njsde = NeuralJSDE()

        # Option 1: Price with pre-set parameters (no training needed)
        price = njsde.price(spot, strike, T, r, q, sigma,
                            features={'vix': 15.0, 'regime_crisis': 0.1, ...})

        # Option 2: Calibrate to market data, then price
        njsde.calibrate(spot, strikes, market_prices, T, r, q, features)
        price = njsde.price(spot, strike, T, r, q, sigma, features)

    API compatible with HestonJumpDiffusionPricer.
    """

    # Default SDE parameter bounds (after sigmoid/softplus transform)
    PARAM_DEFAULTS = {
        'drift_adj': 0.0,       # drift adjustment (centered at 0)
        'vol_mult': 1.0,        # vol multiplier (centered at 1)
        'jump_intensity': 3.0,  # λ in annualized terms
        'jump_mean': -0.02,     # mean jump size (slightly negative)
        'jump_std': 0.03,       # jump size std
    }

    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 50,
        n_centers: int = 15,
        seed: int = 42,
    ):
        self.n_paths = n_paths if n_paths % 2 == 0 else n_paths + 1
        self.n_steps = n_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.network = NeuralParameterNetwork(
            n_features=6, n_centers=n_centers, n_outputs=5
        )
        self.params: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Feature names for consistent ordering
        self.feature_names = [
            'vix_norm',        # VIX / 100
            'regime_crisis',   # P(crisis)
            'regime_normal',   # P(normal)
            'regime_trending', # P(trending)
            'log_moneyness',   # log(K/S)
            'time_to_expiry',  # T in years
        ]

    def _extract_features(self, features: dict, log_m: float = 0.0,
                           T: float = 0.05) -> np.ndarray:
        """Convert feature dict to array."""
        return np.array([
            features.get('vix_norm', features.get('vix', 15.0) / 100.0),
            features.get('regime_crisis', 0.1),
            features.get('regime_normal', 0.6),
            features.get('regime_trending', 0.2),
            log_m,
            T,
        ], dtype=float)

    def _decode_sde_params(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform raw network outputs to valid SDE parameter ranges.

        Uses sigmoid/softplus to enforce constraints:
        - drift_adj ∈ [-0.5, 0.5]  (tanh)
        - vol_mult ∈ [0.3, 3.0]    (sigmoid scaled)
        - λ ∈ [0.1, 50.0]          (softplus)
        - μ_j ∈ [-0.15, 0.05]      (sigmoid scaled)
        - σ_j ∈ [0.005, 0.10]      (sigmoid scaled)
        """
        drift_adj = 0.5 * np.tanh(raw[:, 0])
        vol_mult = 0.3 + 2.7 / (1.0 + np.exp(-raw[:, 1]))
        lam = 0.1 + 49.9 / (1.0 + np.exp(-raw[:, 2]))
        mu_j = -0.15 + 0.20 / (1.0 + np.exp(-raw[:, 3]))
        sig_j = 0.005 + 0.095 / (1.0 + np.exp(-raw[:, 4]))

        return {
            'drift_adj': drift_adj,
            'vol_mult': vol_mult,
            'lambda_j': lam,
            'mu_j': mu_j,
            'sigma_j': sig_j,
        }

    def _simulate(
        self,
        spot: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        sde_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Simulate SDE paths with neural-parameterized dynamics.

        Returns terminal spot values S(T).
        """
        n = self.n_paths
        n_steps = max(int(T * 252), 5) if self.n_steps is None else self.n_steps
        dt = T / n_steps
        half_n = n // 2

        # Extract scalar SDE params (use first element if array)
        drift_adj = float(sde_params['drift_adj'][0]) if len(sde_params['drift_adj']) > 0 else 0.0
        vol_mult = float(sde_params['vol_mult'][0]) if len(sde_params['vol_mult']) > 0 else 1.0
        lam = float(sde_params['lambda_j'][0]) if len(sde_params['lambda_j']) > 0 else 3.0
        mu_j = float(sde_params['mu_j'][0]) if len(sde_params['mu_j']) > 0 else -0.02
        sig_j = float(sde_params['sigma_j'][0]) if len(sde_params['sigma_j']) > 0 else 0.03

        # Effective vol
        sigma_eff = sigma * vol_mult

        # Jump compensator
        k_comp = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0

        # Drift with neural adjustment
        drift = (r - q - lam * k_comp + drift_adj) * dt

        log_S = np.full(n, np.log(max(spot, 1e-8)))

        for step in range(n_steps):
            # Antithetic normals
            z_half = self.rng.standard_normal(half_n)
            z = np.concatenate([z_half, -z_half])

            # Jumps
            n_jumps_half = self.rng.poisson(lam * dt, half_n)
            n_jumps = np.concatenate([n_jumps_half, n_jumps_half])
            z_jump = np.concatenate([
                self.rng.standard_normal(half_n),
                -self.rng.standard_normal(half_n),
            ])
            jump_sizes = np.where(
                n_jumps > 0,
                n_jumps * mu_j + np.sqrt(np.maximum(n_jumps, 1e-8)) * sig_j * z_jump,
                0.0,
            )

            log_S += drift - 0.5 * sigma_eff ** 2 * dt + sigma_eff * np.sqrt(dt) * z + jump_sizes

        return np.exp(log_S)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def price(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float = 0.065,
        q: float = 0.012,
        sigma: float = 0.15,
        option_type: str = 'CE',
        features: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Price an option using Neural J-SDE.

        Returns (price, std_error).
        """
        if features is None:
            features = {}

        # Get SDE parameters from neural network
        log_m = np.log(max(strike, 1e-8) / max(spot, 1e-8))
        X = self._extract_features(features, log_m, T).reshape(1, -1)

        if self.params is not None:
            raw = self.network.forward(X, self.params)
        else:
            # Default params: initialize network with zeros → defaults
            raw = np.zeros((1, 5))

        sde_params = self._decode_sde_params(raw)

        # Reset RNG for reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Simulate
        S_T = self._simulate(spot, T, r, q, sigma, sde_params)

        # Payoffs
        disc = np.exp(-r * T)
        if option_type.upper() in ('CE', 'CALL'):
            payoffs = np.maximum(S_T - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)

        price_val = float(disc * np.mean(payoffs))
        std_err = float(disc * np.std(payoffs) / np.sqrt(len(payoffs)))

        return max(price_val, 0.0), std_err

    def calibrate(
        self,
        spot: float,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T: float,
        r: float = 0.065,
        q: float = 0.012,
        sigma: float = 0.15,
        option_type: str = 'CE',
        features: Optional[dict] = None,
        max_iter: int = 50,
    ) -> Dict:
        """
        Calibrate Neural J-SDE to market prices.

        Learns network weights θ to minimize:
            Σ_i |Model(K_i; θ) - Market(K_i)|²
        """
        if features is None:
            features = {}

        strikes = np.asarray(strikes, dtype=float)
        market_prices = np.asarray(market_prices, dtype=float)

        # Initialize params
        if self.params is None:
            rng = np.random.default_rng(42)
            self.params = rng.normal(0, 0.1, self.network.n_params)

        n_paths_orig = self.n_paths
        self.n_paths = min(self.n_paths, 2000)  # reduce for calibration speed

        def loss(params):
            total = 0.0
            for K, mkt_p in zip(strikes, market_prices):
                log_m = np.log(K / spot)
                X = self._extract_features(features, log_m, T).reshape(1, -1)
                raw = self.network.forward(X, params)
                sde_params = self._decode_sde_params(raw)
                self.rng = np.random.default_rng(self.seed)
                S_T = self._simulate(spot, T, r, q, sigma, sde_params)
                disc = np.exp(-r * T)
                if option_type.upper() in ('CE', 'CALL'):
                    payoffs = np.maximum(S_T - K, 0)
                else:
                    payoffs = np.maximum(K - S_T, 0)
                model_p = disc * np.mean(payoffs)
                total += (model_p - mkt_p) ** 2
            # L2 regularization
            total += 0.001 * np.sum(params ** 2)
            return float(total)

        try:
            result = minimize(
                loss, self.params,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-4},
            )
            self.params = result.x
            self.is_calibrated = True
            final_loss = result.fun
        except Exception:
            final_loss = float('inf')

        self.n_paths = n_paths_orig

        return {
            'loss': float(final_loss),
            'n_strikes': len(strikes),
            'is_calibrated': self.is_calibrated,
        }

    def get_learned_dynamics(self, features: dict, T: float = 0.05) -> Dict[str, float]:
        """
        Inspect the learned SDE parameters for given market state.

        Returns the human-readable drift, vol, jump parameters
        that the network would use for pricing.
        """
        X = self._extract_features(features, 0.0, T).reshape(1, -1)
        if self.params is not None:
            raw = self.network.forward(X, self.params)
        else:
            raw = np.zeros((1, 5))

        decoded = self._decode_sde_params(raw)
        return {
            'drift_adjustment': float(decoded['drift_adj'][0]),
            'vol_multiplier': float(decoded['vol_mult'][0]),
            'jump_intensity': float(decoded['lambda_j'][0]),
            'jump_mean': float(decoded['mu_j'][0]),
            'jump_std': float(decoded['sigma_j'][0]),
        }
