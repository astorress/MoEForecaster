# 🔋 Mixture of Experts (MoE) para Pronóstico de Demanda Eléctrica del SIN Colombia

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](#)

> **Sistema de pronóstico de demanda eléctrica horaria del SIN (Sistema Interconectado Nacional) de Colombia basado en Mixture of Experts (MoE).**

---

## 📑 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de Notebooks](#pipeline-de-notebooks)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Resultados y Métricas](#resultados-y-métricas)
- [Estructura de Datos](#estructura-de-datos)
- [Documentación Detallada](#documentación-detallada)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## 🎯 Descripción General

### Problema

El **Sistema Interconectado Nacional (SIN)** de Colombia requiere pronósticos precisos de demanda eléctrica en horizontes de corto plazo (24h a 1 semana) para:

- Optimización del despacho de generación
- Planificación de mantenimiento
- Reducción de costos operacionales

La demanda presenta **múltiples patrones simultáneos**:

- **Diario (24h)**: ciclos intradía (madrugada baja, mediodía pico, noche media)
- **Semanal (168h)**: diferencia laboral vs. fines de semana
- **Estacional**: variaciones mensuales y anuales

### Solución: Mixture of Experts (MoE)

Un pronóstico adaptativo que:

1. **Identifica dominios** mediante clustering (perfiles diarios × perfiles semanales)
2. **Especifica expertos** — un modelo optimizado por dominio
3. **asignación dinámica de expertos** — elige el experto más apropiado según el contexto actual

**Resultado:** Mayor precisión en escenarios complejos (feriados, anomalías, cambios de patrón).

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS HORARIOS (52,608)                  │
│              API SINERGOX (XM S.A.) 2020-2025               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │  01: ADQUISICIÓN & EDA        │
         │  - Descarga API               │
         │  - Validación                 │
         │  - Análisis exploratorio      │
         └────────────────┬──────────────┘
                          ↓
         ┌───────────────────────────────┐
         │  02: PREPROCESAMIENTO         │
         │  - Outlier detection (STL)    │
         │  - Imputación adaptativa      │
         │  - 38 features engineer       │
         └────────────────┬──────────────┘
                          ↓
         ┌───────────────────────────────────────────────┐
         │  03: CLUSTERING                               │
         │  ├─ TYPE_DAY: perfiles 24h (K-means)          │
         │  └─ TYPE_WEEK: perfiles 168h (K-means)        │
         │     → Cartesian product: (type_day, type_week)│
         └────────────────┬──────────────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  04: ANÁLISIS & CLASIFICADORES       │
         │  - Validar clusters vs. calendario   │
         │  - Entrenar clasificadores           │
         │    (LogReg, RF, SVM, GradBoost)      │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  05: SELECCIÓN DE EXPERTOS           │
         │  - 5 familias × 5 expertos           │
         │  - Walk-forward validation           │
         │  - Homogéneo vs. Heterogéneo         │
         │  → 25 modelos finales                │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  06: DISEÑO DEL GATING               │
         │  - Hard gating (baseline)            │
         │  - Soft (distancia, probabilístico)  │
         │  - Adaptativo (confianza)            │
         │  → Mejor mecanismo seleccionado      │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  07: OPTIMIZACIÓN HIPERPARÁMETROS    │
         │  - Random Search + Successive Halving│
         │  - HPO por experto                   │
         │  - HPO del gating                    │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  08: ENSAMBLE PRODUCCIÓN             │
         │  - Reentrenamiento completo          │
         │  - Clase MoEForecaster               │
         │  - Serialización (.pkl)              │
         │  → Modelo listo para producción      │
         └────────────────┬─────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────┐
    │         EVALUACIÓN Y VALIDACIÓN                │
    ├────────────────────────────────────────────────┤
    │  9: Benchmark comparativo                      │
    │      - MoE vs. 30 modelos (3 familias)         │
    │      - Análisis (desempeño, eficiencia,        │
    │        interpretabilidad, robustez)            │
    └────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
metodologia_moe/
├── README.md                          ← Este archivo
├── RESUMEN_NOTEBOOKS.md               ← Descripción detallada de cada notebook
│
├── data/                             ← Data utilizada
│   ├── classifiers/
│   ├── clustering/
│   ├── processed/
│   └── raw/
│
├── models/                           ← Modelos entrenados
│   ├── clustering/
│   ├── classifiers/
│   ├── experts/
│   ├── gating/
│   ├── hyperparameters/
│   └── production/
│
├── results/                           ← Gráficos y reportes
│
└── notebooks/                         ← Pipeline Jupyter
    ├── 01_data_acquisition.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_clustering.ipynb
    ├── 04_cluster_analysis_classifiers.ipynb
    ├── 05_expert_selection.ipynb
    ├── 06_gating_design.ipynb
    ├── 07_hpo.ipynb
    ├── 08_production_assembly.ipynb
    └── 09_benchmark_comparison.ipynb
```

---

## 🔄 Pipeline de Notebooks

Cada notebook sigue la estructura estandarizada:

```
# Notebook NN -- Nombre

---

## Objetivo
[Qué hace el notebook]

## Entrada
[Datos/archivos que consume]

## Salida
[Archivos que genera]

---

## Instalación
## Configuración
## Imports
## Carga de Datos
## [Análisis Principal]
## Guardado de Archivos
## Resumen
```

### Flujo Secuencial

| NB     | Nombre              | Objetivo                          | Entrada                | Salida                 |
| ------ | ------------------- | --------------------------------- | ---------------------- | ---------------------- |
| **01** | Adquisición         | Descargar 5 años del SIN          | API SINERGOX           | `demand.parquet`       |
| **02** | Preprocesamiento    | Outliers, imputación, 38 features | `demand.parquet`       | `features.parquet`     |
| **03** | Clustering Dual     | TYPE_DAY + TYPE_WEEK              | `features.parquet`     | Asignaciones cluster   |
| **04** | Análisis Clusters   | Validar clusters vs. calendario   | features + clusters    | Clasificadores         |
| **05** | Selección Expertos  | 25 modelos (5×5)                  | features + classifiers | Modelos expertos       |
| **06** | Gating Design       | Hard/Soft/Adaptativo              | features + expertos    | Mejor gating           |
| **07** | HPO                 | Optimizar hiperparámetros         | features + modelos     | Parámetros optimizados |
| **08** | Ensamble Producción | MoEForecaster final               | features + HPO params  | `.pkl` + API           |
| **09** | Benchmark           | MoE vs. 30 modelos                | features + MoE         | Ranking exhaustivo     |

---

## 📦 Requisitos e Instalación

### Requisitos del Sistema

#### Hardware Recomendado

| Componente         | Especificación                 |
| ------------------ | ------------------------------ |
| **GPU**            | NVIDIA Tesla T4 (16GB VRAM)    |
| **CPU**            | 4+ cores                       |
| **RAM**            | 16GB mínimo (32GB recomendado) |
| **Almacenamiento** | 5GB libre                      |

#### Software

| Componente | Versión                  |
| ---------- | ------------------------ |
| **Python** | 3.12.13                  |
| **CUDA**   | 12.8                     |
| **cuDNN**  | Compatible con CUDA 12.8 |

> **Nota:** Este proyecto está optimizado para **Google Colab** con GPU Tesla T4. Las versiones específicas están validadas en ese entorno.

### Dependencias Principales

```python
# Procesamiento de datos
pandas==2.2.2
numpy==2.0.2
pyarrow>=12.0.0

# Machine Learning
scikit-learn==1.6.1
xgboost==3.2.0
lightgbm==4.6.0
statsmodels==0.14.6
scikit-optimize>=0.9.0

# Deep Learning (opcional)
torch>=2.10.0  # Con CUDA 12.8

# Series temporales y análisis
scipy==1.16.3
holidays>=0.33

# Visualización
matplotlib==3.10.0
seaborn==0.13.2
plotly>=5.16.0

# Utilidades
joblib>=1.3.0
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.66.0

# Jupyter
jupyter>=1.0.0
notebook>=7.0.0
jupyterlab>=4.0.0
```

### Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/usuario/metodologia_moe.git
cd metodologia_moe

# 2. Crear ambiente virtual (recomendado)
python3.12 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar Jupyter
jupyter notebook notebooks/
```

### Ejecución en Google Colab (Recomendado)

```python
# En la primera celda de Colab:
!pip install -q -r https://raw.githubusercontent.com/usuario/metodologia_moe/main/requirements.txt

# Los notebooks están optimizados para ejecutarse en Colab con GPU
# Habilita GPU: Menú > Entorno de ejecución > Cambiar tipo de entorno > GPU
```

### Archivo `requirements.txt`

Especificaciones exactas validadas en Google Colab con Tesla T4:

```
# Procesamiento de datos
pandas==2.2.2
numpy==2.0.2
pyarrow==12.0.0

# Machine Learning
scikit-learn==1.6.1
xgboost==3.2.0
lightgbm==4.6.0
statsmodels==0.14.6
scikit-optimize>=0.9.0

# Series temporales
scipy==1.16.3
holidays>=0.33

# Visualización
matplotlib==3.10.0
seaborn==0.13.2
plotly>=5.16.0

# Serialización
joblib>=1.3.0

# Jupyter
jupyter>=1.0.0
notebook>=7.0.0
jupyterlab>=4.0.0

# Utilidades
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.66.0
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Ejecutar el Pipeline Completo (Recomendado)

```bash
# Navega a la carpeta de notebooks
cd notebooks/

# Abre Jupyter
jupyter notebook

# Ejecuta los notebooks en orden:
# 01_data_acquisition.ipynb → 02_preprocessing.ipynb → ... → 10_benchmark_comparison.ipynb

# Tiempo total estimado: 4-6 horas (depende de tu máquina)
```

### Opción 2: Usar el Modelo Entrenado (Rápido)

```python
import pickle
import pandas as pd

# Cargar modelo entrenado
with open('models/production/moe_forecaster.pkl', 'rb') as f:
    moe = pickle.load(f)

# Cargar últimos datos
features = pd.read_parquet('data/processed/features.parquet')

# Pronóstico 168h (1 semana)
forecast = moe.predict(h=168)

# Obtener pesos de expertos
weights = moe.get_expert_weights()

print(f"Pronóstico MAE: {forecast['mae']:.2f} kWh")
print(f"Expertos asignados:\n{weights}")
```

### Opción 3: Ejecutar Solo Notebook Específico

```bash
# Ej: Ejecutar solo evaluación sin reentrenar modelos
jupyter notebook notebooks/09_evaluation.ipynb
```

---

## 📖 Documentación Detallada

Para más información, consulta:

- **[RESUMEN_NOTEBOOKS.md](RESUMEN_NOTEBOOKS.md)** — Descripción completa de cada notebook y su contenido
- **Notebooks individuales** — Código fuente y comentarios detallados

---

## 🤝 Contribuciones

Este proyecto es de código abierto. Para contribuir:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m "Agregar mejora"`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 👨‍💼 Autor

**Sebastián Torres**

- Email: [anderson.torress@udea.edu.co](mailto:anderson.torress@udea.edu.co)
- GitHub: [@usuario](https://github.com/astorress)

## 🔗 Enlaces Útiles

- [Documentación de XM](https://www.xm.com.co/)
- [SIN - Operación del sistema](https://www.xm.com.co)

---

**Última actualización:** 3 de mayo de 2026  
**Versión:** 1.0.0
