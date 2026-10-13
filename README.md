# Segundo parcial – Introducción al Machine Learning (2026-2)

Universidad El Bosque – Estadística.


**Autores:** <Mariana Lizarazo> y <Karol Salcedo>

## Estructura

- `data/`: archivos `train_A.csv`, `train_B.csv`, `train_C.csv` y `validation.csv`.
- `src/parcial_ml/data.py`: carga de los datos.
- `src/parcial_ml/models.py`: ajuste y predicción de regresiones polinomiales.
- `src/parcial_ml/risk.py`: riesgo empírico con pérdida cuadrática.
- `src/parcial_ml/analysis.py`: barrido de grados, malla común, hipótesis promedio,
  varianza V_d y riesgo promedio de validación.
- `notebooks/parcial.ipynb`: cuaderno con las tablas, figuras y respuestas.
- `resultados/`: tablas (.csv) y figuras (.png) generadas por el cuaderno.

## Instalación

Requiere Python 3.9 o superior. Desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
pip install -e .
```

## Cómo reproducir los resultados

1. Abrir `notebooks/parcial.ipynb` en VS Code o Jupyter.
2. Seleccionar el kernel de Python donde se instaló el proyecto.
3. Ejecutar el cuaderno completo (Restart y Run All).

Las tablas y figuras se guardan en `resultados/`.

## Notas

- El archivo `validation.csv` no se usa para ajustar ningún modelo.
- En `models.py` la variable x se divide entre 2.5 antes de generar las potencias,
  para evitar inestabilidad numérica con grados altos. No cambia el modelo, que
  sigue siendo un polinomio de grado d.