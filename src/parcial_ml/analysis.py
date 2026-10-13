import pandas as pd
import numpy as np

from .data import load_dataset
from .models import fit_poly, predict_poly
from .risk import empirical_risk

#ejercicio1

def degree_sweep(train_name="train_A", val_name="validation", degrees=range(1, 16)):
    """Ejercicio 1: riesgo de entrenamiento y de validación para cada grado."""
    x_tr, y_tr = load_dataset(train_name)
    x_val, y_val = load_dataset(val_name)
    filas = []
    for d in degrees:
        modelo = fit_poly(x_tr, y_tr, d)
        filas.append({
            "grado": d,
            "riesgo_train": empirical_risk(y_tr, predict_poly(modelo, x_tr)),
            "riesgo_val": empirical_risk(y_val, predict_poly(modelo, x_val)),
        })
    return pd.DataFrame(filas)


def best_degrees(tabla):
    """Devuelve (d_train, R_train, d_val, R_val), los grados que minimizan cada riesgo."""
    i = tabla["riesgo_train"].idxmin()
    j = tabla["riesgo_val"].idxmin()
    return (int(tabla.loc[i, "grado"]), float(tabla.loc[i, "riesgo_train"]),
            int(tabla.loc[j, "grado"]), float(tabla.loc[j, "riesgo_val"]))

SETS = ("train_A", "train_B", "train_C")
DEGREES_EX2 = (2, 6, 15)


#ejercicio2

def fit_all(degrees=DEGREES_EX2, sets=SETS):
    """Ajusta un modelo por cada (conjunto, grado). Devuelve {(conjunto, grado): modelo}."""
    modelos = {}
    for s in sets:
        x, y = load_dataset(s)
        for d in degrees:
            modelos[(s, d)] = fit_poly(x, y, d)
    return modelos


def make_grid(m=300, lo=-2.5, hi=2.5):
    """Malla de m puntos igualmente espaciados en [lo, hi]."""
    return np.linspace(lo, hi, m)


def grid_predictions(modelos, malla, degrees=DEGREES_EX2, sets=SETS):
    """Predicciones en la malla: {grado: arreglo de forma (3, m)}, una fila por conjunto."""
    return {d: np.vstack([predict_poly(modelos[(s, d)], malla) for s in sets])
            for d in degrees}


def mean_hypothesis_and_variance(preds):
    """Devuelve ({grado: h_bar_d en la malla}, {grado: V_d})."""
    hbar = {d: p.mean(axis=0) for d, p in preds.items()}
    var = {d: float(np.mean(np.mean((p - hbar[d]) ** 2, axis=0)))
           for d, p in preds.items()}
    return hbar, var


#ejercicio3

def mean_validation_risk(modelos, degrees=DEGREES_EX2, sets=SETS, val_name="validation"):
    """R_val(d): promedio, sobre los tres conjuntos, del riesgo de validación."""
    x_val, y_val = load_dataset(val_name)
    return {d: float(np.mean([empirical_risk(y_val, predict_poly(modelos[(s, d)], x_val))
                              for s in sets]))
            for d in degrees}


def summary_table(rval, var):
    """Tabla del Ejercicio 3: grado, R_val(d) y V_d."""
    return pd.DataFrame({"grado": list(rval),
                         "Rval": list(rval.values()),
                         "Vd": [var[d] for d in rval]})