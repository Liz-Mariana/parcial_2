import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dominio de x en el experimento: [-2.5, 2.5]. Se reescala a [-1, 1]
# para que las potencias altas no se disparen numéricamente.
X_SCALE = 2.5


class PolyModel:
    """Regresión polinomial de grado `degree`."""

    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.reg = LinearRegression()

    def fit(self, x, y):
        z = np.asarray(x, dtype=float).reshape(-1, 1) / X_SCALE
        self.reg.fit(self.poly.fit_transform(z), y)
        return self

    def predict(self, x):
        z = np.asarray(x, dtype=float).reshape(-1, 1) / X_SCALE
        return self.reg.predict(self.poly.transform(z))


def fit_poly(x, y, degree):
    """Ajusta un polinomio de grado `degree` y devuelve el modelo."""
    return PolyModel(degree).fit(x, y)


def predict_poly(model, x):
    """Predice con un modelo ya ajustado."""
    return model.predict(x)