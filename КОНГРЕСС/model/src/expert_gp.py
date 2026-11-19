# -*- coding: utf-8 -*-
"""Layer 2a · Expert Score — Gaussian Process regression (small-data, uncertainty).
Использует экспертные баллы из sites.csv и predicts 'expert_score' для новых площадок
(через RBF kernel на 6 features: deficit, oyat, logistics, industry, economy, geo_climate).
Возвращает mu ± sigma.
"""
import numpy as np
from typing import Tuple

class ExpertGPR:
    """Упрощённый GPR на sklearn.gaussian_process (если установлен), иначе — линейный proxy."""
    def __init__(self):
        self.X_train=None; self.y_train=None; self.scaler=None
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            self.GP_CLS=GaussianProcessRegressor
            self.KERNEL=ConstantKernel(1.0)*(RBF(length_scale=1.0))
            self._use_sklearn=True
        except ImportError:
            self._use_sklearn=False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train=X.copy(); self.y_train=y.copy()
        if self._use_sklearn:
            self.gp=self.GP_CLS(kernel=self.KERNEL, alpha=1e-6, n_restarts_optimizer=3)
            self.gp.fit(X,y)

    def predict_site(self, x: np.ndarray) -> Tuple[float,float]:
        # x: array-like of length 6
        x=np.asarray(x, dtype=float).reshape(1,-1)
        # Normalize to [0,1] using train min/max
        xmin=self.X_train.min(axis=0); xmax=self.X_train.max(axis=0)+1e-9
        xn=(x-xmin)/(xmax-xmin)
        if self._use_sklearn and self.X_train is not None:
            mu,sigma=self.gp.predict(xn,sigma=True)
            return float(mu[0]), float(sigma[0])
        # Fallback linear proxy: weighted average
        w=np.array([0.20,0.08,0.20,0.12,0.25,0.15])
        mu=float(w.dot(xn[0])*25.0)  # scale to original expert_score range ~15-25
        return max(mu, 0.0), 2.0  # sigma fixed for safety
