import numpy as np


def empirical_risk(y_true, y_pred):
    """Riesgo empírico con pérdida cuadrática: (1/n) * suma((y_i - h(x_i))**2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))