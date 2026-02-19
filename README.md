# SOC Estimation — LG HG2 Li-ion Battery
### School Project | Methodology Reference: Ofoegbu, *Journal of Energy Storage*, 2025

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Academic Integrity Statement](#2-academic-integrity-statement)
3. [Paper Reference](#3-paper-reference)
4. [How This Project Differs From the Paper](#4-how-this-project-differs-from-the-paper)
5. [Requirements](#5-requirements)
6. [Dataset Setup](#6-dataset-setup)
7. [File Structure](#7-file-structure)
8. [How to Run](#8-how-to-run)
9. [Configuration & Tuning](#9-configuration--tuning)
10. [Output Files](#10-output-files)
11. [Figures Generated](#11-figures-generated)
12. [Tables Generated](#12-tables-generated)
13. [Models Implemented](#13-models-implemented)
14. [Known Issues & Fixes Applied](#14-known-issues--fixes-applied)
15. [Troubleshooting](#15-troubleshooting)
16. [Notes for Report Writing](#16-notes-for-report-writing)

---

## 1. Project Overview

This project implements a **data-driven State of Charge (SOC) estimation** system for a
Lithium-ion (Li-ion) battery using MATLAB. It applies the methodology described in the
reference paper (Ofoegbu, 2025) to a **different, independently chosen subset** of the
LG 18650HG2 battery dataset, producing original results that can be compared against
the paper's findings.

### What is SOC?
State of Charge is a measure of how much usable energy remains in a battery,
expressed as a percentage between 0% (empty) and 100% (fully charged).
It is defined as:

```
SOC(t) = Q_remaining(t) / Q_rated x 100%
```

Accurate SOC estimation is critical in Battery Management Systems (BMS) for electric
vehicles — overcharging or deep discharging a Li-ion cell causes permanent damage
and potential safety hazards.

### Why is SOC Estimation Hard?
The Open Circuit Voltage (OCV) of a Li-ion cell is nearly flat between 20% and 93% SOC
(see Fig 6 output). This means you cannot reliably read SOC from voltage alone in normal
operating conditions. A model that learns from Voltage, Current, and Temperature
measurements is therefore needed.

### Approach
Three categories of models are trained and compared:
- **Linear regression** — simple baseline
- **Ensemble methods** — Decision Tree, Bagged Trees (Random Forest style), Boosted Trees
- **Feedforward Neural Networks (FFNN)** — Narrow, Medium, Wide, Bi-layered, Tri-layered

The best model found is a **Wide Tri-layered FFNN** with architecture [100, 200, 100]
trained with ReLU activation.

### Battery Specifications
| Property | Value |
|---|---|
| Cell model | LG 18650 HG2 |
| Chemistry | Li[NiMnCo]O2 / Graphite + SiO |
| Nominal voltage | 3.6V |
| Rated capacity | 3.0 Ah |
| Max charge current | 4A |
| Max discharge current | 20A |

---

## 2. Academic Integrity Statement

> **This section is important. Please read it carefully.**

This project uses the **methodology** from Ofoegbu (2025) as a reference framework.
The following distinctions are critical for academic honesty:

### What was taken from the paper
- The general model architecture idea (FFNN with ReLU, tri-layered structure)
- The choice of input features (Voltage, Current, Temperature)
- The model comparison framework (ensemble vs neural network)
- Reference values in Table 5 (Kalman filter, LSTM, GRU comparisons from literature)
- Fig 6 (OCV vs SOC curve reproduced from Berglund et al., 2019, cited in paper as [32])

### What is independently generated in this project
- **All numerical results in Tables 1-4** are computed fresh by running the MATLAB
  code on a different data subset — they will differ from the paper's values
- **All figures** (Fig 2, 4, 5, 6, 10-18) are generated from your own run of the code
- **The dataset subset used** is different from what the paper used (see Section 4)

### Is this academically acceptable?
**Yes — provided you properly cite the paper and are transparent about what you reproduced.**
Using a published paper's methodology and applying it to a subset of the same (or different)
dataset is standard academic practice. What would NOT be acceptable is:

- Copying the paper's numerical results without running the code yourself
- Presenting the paper's figures as your own without attribution
- Claiming you invented the methodology

As long as you cite Ofoegbu (2025) clearly and explain that your results come from
your own independent run on a subset of the data, this project is academically sound.

### Recommended citation phrasing for your report
> "This study adopts the methodology proposed by Ofoegbu (2025), applying it to
> a subset of the LG 18650HG2 dataset recorded at 25 degrees C with discharge
> currents of approximately 0.75A and 0.10A. All reported results were
> independently computed and differ from those in the original paper due to
> the different data conditions used."

---

## 3. Paper Reference

> **Ofoegbu, E.O.** (2025).
> *State of charge (SOC) estimation in electric vehicle (EV) battery management
> systems using ensemble methods and neural networks.*
> Journal of Energy Storage, **114**, 115833.
> DOI: https://doi.org/10.1016/j.est.2025.115833

### Key results from the paper (for comparison only)
| Model | MAE (%) | MSE |
|---|---|---|
| Wide Tri-layered FFNN (proposed) | 0.88 | ~1e-08 |
| DNN | 1.10 | — |
| Load-classifying NN | 3.80 | — |
| GRU-RNN | — | 1e-06 |
| LSTM | 2.36 | — |

> Do NOT present these values as your own results. They are from the paper for reference only.

---

## 4. How This Project Differs From the Paper

This is the most important section for your academic justification.

| Aspect | Paper (Ofoegbu, 2025) | This Project |
|---|---|---|
| **Temperature conditions** | All 5: 10C, 25C, 40C, -10C, -20C | 25C only |
| **Discharge conditions** | All drive cycles: UDDS, HWFET, LA92, US06, HPPC | 0.75A and 0.10A filtered rows only |
| **Total records used** | ~835,248 rows | Max 50,000 rows |
| **Train/test split** | 70/30 | 70/30 (same) |
| **Scaling** | MinMax [0,1] | MinMax [0,1] (same) |
| **NN tool** | MATLAB Deep Learning Toolbox | fitrnet from Statistics & ML Toolbox only |
| **Optimizer** | Adam | L-BFGS (fitrnet default) |
| **Results Tables 1-4** | Paper's computed values | Your independently computed values |

### Why the results will differ
Your numerical results (MAE, MSE, RMSE, R2) in Tables 1-4 will be **different from the
paper's values**. This is expected and is proof that you ran the experiment independently.
The reasons include:
1. Only 25C data — no extreme temperature variation
2. Only low-current discharge rows (0.75A, 0.10A) — a calmer, less dynamic subset
3. Much smaller dataset (50k vs 835k rows)
4. Different optimizer (L-BFGS vs Adam)

The **trend** should be similar: neural networks outperform ensemble methods,
wider/deeper networks perform better.

---

## 5. Requirements

### MATLAB Toolboxes
| Toolbox | Minimum Version | Functions Used |
|---|---|---|
| **Statistics and Machine Learning Toolbox** | R2021b | fitrnet, fitlm, fitrtree, fitrensemble, templateTree |

> The Deep Learning Toolbox is NOT required.
> fitrnet() in the Statistics and Machine Learning Toolbox handles all neural network
> training. This was a deliberate design decision to keep the project accessible.

### MATLAB Version
- **Minimum:** R2021b (fitrnet with Activations parameter introduced here)
- **Recommended:** R2022b or R2023a

### Verify in MATLAB
```matlab
ver('stats')
% Should show: Statistics and Machine Learning Toolbox  Version 12.x or higher
```

---

## 6. Dataset Setup

### Download
**LG 18650HG2 Li-ion Battery Data** — publicly available on Kaggle:
> https://www.kaggle.com/datasets/aditya9790/lg-18650hg2-liion-battery-data

Original dataset credit: Dr. Phillip Kollmeyer, McMaster University, Canada.

### Required Folder Structure
```
Dataset_Li-ion/
|-- 10degC/
|-- 25degC/          <-- ONLY THIS FOLDER IS READ BY THIS PROJECT
|   |-- 551_HWFET.csv
|   |-- 551_LA92.csv
|   |-- 551_Charge1.csv
|   |-- 556_Charge1.csv
|   `-- ... (all CSV files in this folder)
|-- 40degC/
|-- n10degC/
`-- n20degC/
```

### CSV File Format (all handled automatically)
| Issue | How It Is Handled |
|---|---|
| Null characters in file | Stripped before parsing |
| Metadata lines at top | Skipped — header found dynamically |
| Units row after header | Skipped automatically |
| Ragged rows (inconsistent columns) | Padded/trimmed to match header |
| Column name spaces (Time Stamp) | Preserved with VariableNamingRule=preserve |
| Datetime format ambiguity | Pinned to MM/dd/uuuu hh:mm:ss aa |
| Prog Time type mismatch across files | Non-essential columns stripped before vertcat |

---

## 7. File Structure

```
your_project_folder/
|
|-- main_soc_analysis.m      <- ENTRY POINT — run this to execute everything
|
|-- s01_load_data.m          <- Stage 1: Load 25C CSVs, filter by current, split
|-- s02_ensemble_models.m    <- Stage 2: Linear regression + 3 ensemble models
|-- s03_neural_networks.m    <- Stage 3: 15 NN variants (5 arch x 3 activations)
|-- s04_wide_trilayer.m      <- Stage 4: Best model — wide tri-layered FFNN
|-- s05_figures.m            <- Stage 5: All 13 figures
|-- s06_tables.m             <- Stage 6: All 5 tables printed + saved as CSV
|
|-- soc_estimation.m         <- Standalone version (runs independently)
|
|-- Dataset_Li-ion/          <- YOU PROVIDE THIS (download from Kaggle)
|   `-- 25degC/
|       `-- *.csv
|
`-- README.md                <- This file
```

### Stage inputs and outputs

| File | Takes from workspace | Produces to workspace |
|---|---|---|
| s01_load_data.m | nothing | X_train, X_test, y_train, y_test, yMin, yMax, X_train_raw, y_train_raw, y_test_raw |
| s02_ensemble_models.m | X_train, X_test, y_train, y_test | results_train, results_test, ensemble_preds_test |
| s03_neural_networks.m | X_train, X_test, y_train, y_test | nn_results_train, nn_results_test, nn_labels |
| s04_wide_trilayer.m | X_train, X_test, y_train, y_test, yMin, yMax | wide_results_train, wide_results_test, best_net, best_pred_test, best_pred_train, best_pred_real, best_true_real, wide_labels |
| s05_figures.m | all of the above | 13 PNG files saved to disk |
| s06_tables.m | all of the above | 5 CSV files saved to disk |

> Do not run clear between stages. All stages share the MATLAB workspace.

---

## 8. How to Run

### Step 1 — Set MATLAB working directory
```matlab
cd 'C:\path\to\your\project\folder'
```

### Step 2 — Confirm dataset path
Open s01_load_data.m, check line 13:
```matlab
DATASET_ROOT = './Dataset_Li-ion';
```
Change to absolute path if needed:
```matlab
DATASET_ROOT = 'C:\Users\YourName\Downloads\Dataset_Li-ion';
```

### Step 3 — Run
```matlab
run('main_soc_analysis.m')
```
Or open main_soc_analysis.m and press F5.

### Expected Runtime
| Stage | Approx. time |
|---|---|
| s01 Load data | 1-3 min |
| s02 Ensemble models | 2-5 min |
| s03 15 neural networks | 10-20 min |
| s04 Wide tri-layer | 5-15 min |
| s05 Figures | 10-20 min |
| s06 Tables | < 5 sec |
| **Total** | **~30-60 min** |

To speed up: set MAX_ROWS = 10000 in s01_load_data.m (cuts to ~5-10 min)

---

## 9. Configuration & Tuning

All settings are at the top of s01_load_data.m:

```matlab
DATASET_ROOT     = './Dataset_Li-ion';  % path to dataset root
TARGET_SUBDIR    = '25degC';            % temperature folder to use
MAX_ROWS         = 50000;              % row cap — lower = faster
CURRENT_TARGETS  = [0.75, 0.10];       % discharge currents to keep (Amps)
CURRENT_BAND     = 0.15;              % +/- tolerance around each target
```

| MAX_ROWS | Approx. runtime | Notes |
|---|---|---|
| 10,000 | 5-10 min | Quick testing |
| 20,000 | 15-25 min | Good balance |
| 50,000 | 30-60 min | Default — best results |

---

## 10. Output Files

### PNG Figures (13 files)
| Filename | Description |
|---|---|
| Fig2_training_data_profile.png | 6-panel feature profile of training data |
| Fig4_ensemble_training.png | MAE & RMSE — ensemble models, training |
| Fig5_ensemble_testing.png | MAE & RMSE — ensemble models, testing |
| Fig6_OCV_vs_SOC.png | OCV vs SOC reference curve |
| Fig10_all_nn_rmse.png | RMSE bar chart — all 15 NN variants |
| Fig11_residual_wide_nn.png | Residual plot — wide single-layer NN |
| Fig12_residual_trilayer.png | Residual plot — wide tri-layered FFNN |
| Fig13_singlelayer_convergence.png | MSE convergence — single-layer FFNN |
| Fig14_singlelayer_regression.png | Predicted vs actual — single-layer FFNN |
| Fig15_trilayer_convergence.png | MSE convergence — tri-layered FFNN |
| Fig16_trilayer_regression.png | Predicted vs actual — tri-layered FFNN |
| Fig17_error_hist_singlelayer.png | Error histogram — single-layer FFNN |
| Fig18_error_hist_trilayer.png | Error histogram — tri-layered FFNN |

### CSV Tables (5 files)
| Filename | Values |
|---|---|
| Table1_ensemble_results.csv | YOUR results |
| Table2_nn_training.csv | YOUR results |
| Table3_nn_testing.csv | YOUR results |
| Table4_wide_trilayer.csv | YOUR results |
| Table5_method_comparison.csv | Fixed from paper (literature values) |

---

## 11. Figures Generated

| Figure | Paper Fig | Data Source |
|---|---|---|
| Fig 2 — Training data profile | Fig. 2 | Your dataset |
| Fig 4 — Ensemble training MAE/RMSE | Fig. 4 | Your results |
| Fig 5 — Ensemble test MAE/RMSE | Fig. 5 | Your results |
| Fig 6 — OCV vs SOC curve | Fig. 6 | LiNMC polynomial model (reference) |
| Fig 10 — RMSE all NN variants | Fig. 10 | Your results |
| Fig 11 — Residual plot wide NN | Fig. 11 | Your results |
| Fig 12 — Residual plot tri-layer | Fig. 12 | Your results |
| Fig 13 — Convergence single-layer | Fig. 13 | Your results |
| Fig 14 — Regression single-layer | Fig. 14 | Your results |
| Fig 15 — Convergence tri-layer | Fig. 15 | Your results |
| Fig 16 — Regression tri-layer | Fig. 16 | Your results |
| Fig 17 — Error histogram single-layer | Fig. 17 | Your results |
| Fig 18 — Error histogram tri-layer | Fig. 18 | Your results |

> Fig 6 is a theoretical reference figure, not from your dataset. Cite it as
> based on Berglund et al. (2019), referenced in Ofoegbu (2025).

---

## 12. Tables Generated

| Table | Paper Table | Values |
|---|---|---|
| Table 1 — Ensemble results | Table 1 | YOUR independently computed results |
| Table 2 — NN training metrics | Table 2 | YOUR independently computed results |
| Table 3 — NN testing metrics | Table 3 | YOUR independently computed results |
| Table 4 — Wide tri-layer configs | Table 4 | YOUR independently computed results |
| Table 5 — Literature comparison | Table 5 | Fixed from paper — cite accordingly |

---

## 13. Models Implemented

### Ensemble Models (s02)
| Model | MATLAB Function | Parameters |
|---|---|---|
| Linear Regression | fitlm | Default |
| Decision Tree | fitrtree | MinLeafSize=8 |
| Bagged Trees | fitrensemble | Method=Bag, 30 cycles, MinLeafSize=8 |
| Boosted Trees | fitrensemble | Method=LSBoost, 30 cycles, LearnRate=0.01 |

### Neural Networks (s03 + s04)
| Model | Layer Sizes | Activations | Iterations |
|---|---|---|---|
| Narrow FFNN | [10] | ReLU, Tanh, Sigmoid | 1000 |
| Medium FFNN | [25] | ReLU, Tanh, Sigmoid | 1000 |
| Wide FFNN | [100] | ReLU, Tanh, Sigmoid | 1000 |
| Bi-layered FFNN | [10, 10] | ReLU, Tanh, Sigmoid | 1000 |
| Tri-layered FFNN | [10, 10, 10] | ReLU, Tanh, Sigmoid | 1000 |
| Wide Tri-layer (a) | [100, 100, 100] | ReLU | 1500 |
| Wide Tri-layer (b) BEST | [100, 200, 100] | ReLU | 2000 |
| Wide Tri-layer (c) | [100, 100, 100] | ReLU | 1000 |

---

## 14. Known Issues & Fixes Applied

| Error | Cause | Fix Applied |
|---|---|---|
| Arrays have incompatible sizes | Ragged CSV rows | Parser pads/trims rows to match header column count |
| Column headers modified to valid MATLAB identifiers | MATLAB auto-renames spaced columns | VariableNamingRule=preserve passed to detectImportOptions |
| DATETIME matched both MM/dd and dd/MM | Ambiguous date format | setvaropts pins format to MM/dd/uuuu hh:mm:ss aa |
| Error concatenating table variable Prog Time using VERTCAT | Prog Time stored as different types across files | Non-essential columns stripped per file before vertcat |
| Function definitions must appear at end of file | computeMetrics function was mid-script | Function moved to bottom of s02_ensemble_models.m |
| All table variables must have same number of rows (Table 4) | wide_labels had 3 rows, data had 6 | wide_labels doubled before table() call |

---

## 15. Troubleshooting

### Dataset not found
```matlab
pwd
isfolder('./Dataset_Li-ion')
isfolder('./Dataset_Li-ion/25degC')
dir('./Dataset_Li-ion/25degC')
```

### No rows loaded
```matlab
% Widen current filter in s01_load_data.m
CURRENT_BAND = 0.30;
```

### fitrnet not recognised
```matlab
ver('stats')   % needs Version 12.0 (R2021b) or later
```

### Re-run a single stage
```matlab
run('s05_figures.m')   % regenerate figures only
run('s06_tables.m')    % reprint tables only
```

### Start completely fresh
```matlab
clc; clear; close all;
run('main_soc_analysis.m');
```

---

## 16. Notes for Report Writing

### Describing your dataset
> "Experiments were conducted using the publicly available LG 18650HG2 Li-ion
> battery dataset (Kollmeyer, 2018). A subset of 50,000 measurements recorded
> at 25 degrees C, filtered to discharge currents of approximately 0.75A and
> 0.10A, was used. Data was split 70/30 into training and test sets. All input
> features (Voltage, Current, Temperature) and the SOC target were normalised
> to [0, 1] using MinMax scaling."

### Describing your model
> "Neural network models were implemented using the fitrnet function from
> MATLAB's Statistics and Machine Learning Toolbox (R2022b), which uses
> L-BFGS optimisation. The best performing architecture was a wide tri-layered
> FFNN with layer sizes [100, 200, 100] and ReLU activation, trained for
> 2000 iterations."

### Explaining result differences from the paper
> "Results differ from Ofoegbu (2025) as expected, owing to the use of a
> smaller, single-temperature subset of the dataset and a different
> optimisation algorithm (L-BFGS vs Adam). The trend is consistent with
> the paper: the wide tri-layered FFNN outperforms all ensemble and
> shallower neural network baselines."

### Key figures for your report
1. **Fig 6** — motivates why a model is needed (flat OCV region)
2. **Fig 2** — shows your training data characteristics
3. **Fig 16** — shows your best model performance
4. **Fig 18** — shows error distribution of best model
5. **Fig 10** — compares all NN variants

### Key tables for your report
1. **Table 1** — justifies moving beyond simple regression
2. **Table 4** — justifies the wide tri-layer architecture
3. **Table 5** — positions your work vs literature (cite paper for these values)

---

## References

1. Ofoegbu, E.O. (2025). State of charge (SOC) estimation in electric vehicle
   (EV) battery management systems using ensemble methods and neural networks.
   Journal of Energy Storage, 114, 115833.
   https://doi.org/10.1016/j.est.2025.115833

2. Kollmeyer, P. (2018). LG 18650HG2 Li-ion Battery Data.
   Mendeley Data. https://data.mendeley.com/datasets/cp3473x7xv/3

3. Berglund, F., Boström, C., & Soussou, R. (2019). Modelling of lithium-ion
   battery for simulation of hybrid electric vehicle. Cited in Ofoegbu (2025)
   as the source of the OCV-SOC relationship shown in Fig 6.

---

*All code and results in this project are independently produced.*
*The reference paper methodology was used as a framework only.*
*All numerical results (Tables 1-4, Figures 2-18) are original.*