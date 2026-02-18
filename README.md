# SOC Estimation — LG HG2 Li-ion Battery
### School Project | Based on: Ofoegbu, *Journal of Energy Storage*, 2025

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Paper Reference](#2-paper-reference)
3. [Requirements](#3-requirements)
4. [Dataset Setup](#4-dataset-setup)
5. [File Structure](#5-file-structure)
6. [How to Run](#6-how-to-run)
7. [Configuration & Tuning](#7-configuration--tuning)
8. [Output Files](#8-output-files)
9. [Figures Generated](#9-figures-generated)
10. [Tables Generated](#10-tables-generated)
11. [Models Implemented](#11-models-implemented)
12. [Known Issues & Fixes](#12-known-issues--fixes)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

This project implements a **data-driven State of Charge (SOC) estimation** system for a
Lithium-ion battery using MATLAB. It reproduces the methodology, figures, and tables from
the reference paper listed below.

**What is SOC?**
SOC is a measure of how much charge remains in a battery, expressed as a percentage (0–100%).
It is mathematically defined as:

```
SOC = Q_Remaining / Q_Rated
```

Where `Q_Remaining` is the remaining charge and `Q_Rated` is the rated battery capacity.

**Approach:**
Three categories of models are compared:
- Simple linear regression
- Ensemble methods (Decision Trees, Bagged Trees, Boosted Trees)
- Feedforward Neural Networks (FFNN) of varying width and depth

The key finding (reproduced here) is that a **wide, tri-layered FFNN** achieves the best
SOC estimation accuracy with low computational cost — making it suitable for real Battery
Management Systems (BMS).

**Dataset Used:**
- LG 18650 HG2 Li-ion Battery (3 Ah cell)
- Battery chemistry: Li[NiMnCo]O2 (H-NMC) / Graphite + SiO
- Nominal voltage: 3.6V
- Only the **25°C** temperature subfolder is used in this project
- Only rows with discharge current near **0.75A or 0.10A** are used (configurable)
- Maximum row cap: **50,000 rows** (configurable) to keep training fast

---

## 2. Paper Reference

> **Ofoegbu, E.O.** (2025).
> *State of charge (SOC) estimation in electric vehicle (EV) battery management systems
> using ensemble methods and neural networks.*
> Journal of Energy Storage, 114, 115833.
> https://doi.org/10.1016/j.est.2025.115833

**Key results from the paper (for reference):**
| Model | MAE | MSE |
|---|---|---|
| Wide Tri-layered FFNN (proposed) | 0.88% | ~1e-08 |
| GRU-RNN | — | 1e-06 |
| Load-classifying NN | 3.8% | — |
| DNN | 1.10% | — |

---

## 3. Requirements

### MATLAB Toolboxes Required
| Toolbox | Version | Used For |
|---|---|---|
| **Statistics and Machine Learning Toolbox** | R2021b+ | `fitrnet`, `fitlm`, `fitrtree`, `fitrensemble` |

> ⚠️ **The Deep Learning Toolbox is NOT required.**
> All neural network training uses `fitrnet()` from the Statistics and Machine Learning
> Toolbox only.

### MATLAB Version
- **Minimum:** R2021b (when `fitrnet` was introduced)
- **Tested on:** R2022b / R2023a
- `fitrnet` with `'Activations'` parameter requires R2021b or later

### Verify your toolbox in MATLAB:
```matlab
ver('stats')
```
You should see something like: `Statistics and Machine Learning Toolbox  Version 12.x`

---

## 4. Dataset Setup

### Download
The dataset used is the **LG 18650HG2 Li-ion Battery Data** from Kaggle:
- https://www.kaggle.com/datasets/aditya9790/lg-18650hg2-liion-battery-data

### Expected Folder Structure
After downloading and extracting, your dataset folder must look like this:

```
Dataset_Li-ion/
├── 10degC/
│   ├── 571_Mixed1.csv
│   ├── 571_Mixed2.csv
│   └── ...
├── 25degC/          <-- THIS IS THE ONLY FOLDER USED BY THIS PROJECT
│   ├── 556_Charge1.csv
│   ├── 556_Charge2.csv
│   ├── 556_Mixed1.csv
│   └── ...
├── 40degC/
│   └── ...
├── n10degC/
│   └── ...
└── n20degC/
    └── ...
```

> ℹ️ Only the `25degC` subfolder is read. The other temperature folders are ignored.
> This is intentional — it keeps the project fast and manageable as a school project.

### CSV File Format
Each CSV file in the dataset has this structure:
- Several metadata/comment lines at the top (automatically skipped)
- A **header row** containing: `Time Stamp, Voltage, Current, Temperature, ...`
- A **units row** immediately after the header (automatically skipped)
- Data rows from row 3 onwards

The parser handles all of this automatically, including:
- Null character stripping
- Ragged/inconsistent row lengths (padded/trimmed automatically)
- Datetime format ambiguity (`MM/dd/yyyy`)
- Column name preservation (spaces kept intact)

---

## 5. File Structure

```
your_project_folder/
│
├── main_soc_analysis.m     ← START HERE — runs everything
│
├── s01_load_data.m         ← Stage 1: Load & filter dataset
├── s02_ensemble_models.m   ← Stage 2: Linear regression + tree models
├── s03_neural_networks.m   ← Stage 3: All NN variants (Tables 2 & 3)
├── s04_wide_trilayer.m     ← Stage 4: Best model (wide tri-layered FFNN)
├── s05_figures.m           ← Stage 5: Generate all 12 figures
├── s06_tables.m            ← Stage 6: Print & save all 5 tables
│
├── soc_estimation.m        ← Standalone single-file version (alternative)
│
├── Dataset_Li-ion/         ← Your dataset folder (you provide this)
│   ├── 25degC/
│   └── ...
│
└── README.md               ← This file
```

### Role of Each File

#### `main_soc_analysis.m`
The master entry point. Calls all 6 stage scripts in the correct order.
All variables are shared in the MATLAB workspace between stages —
**do not clear the workspace between stages**.

#### `s01_load_data.m`
- Reads CSV files from `Dataset_Li-ion/25degC/`
- Filters rows to discharge current near **0.75A** or **0.10A**
- Applies MinMax scaling to features (Voltage, Current, Temperature) and target (SOC)
- Performs a **70/30 train/test split** (matching the paper)
- Outputs workspace variables: `X_train`, `X_test`, `y_train`, `y_test`,
  `X_train_raw`, `y_train_raw`, `y_test`, `yMin`, `yMax`, `targetCol`

#### `s02_ensemble_models.m`
- Trains: Linear Regression, Decision Tree, Bagged Trees, Boosted Trees
- Computes MAE, MSE, RMSE, R² for train and test sets
- Outputs: `results_train`, `results_test`, `ensemble_preds_test`

#### `s03_neural_networks.m`
- Trains 15 neural networks: 5 architectures × 3 activations (ReLU, Tanh, Sigmoid)
- Architectures: Narrow[10], Medium[25], Wide[100], Bi-layer[10,10], Tri-layer[10,10,10]
- Outputs: `nn_results_train`, `nn_results_test`, `nn_labels`

#### `s04_wide_trilayer.m`
- Trains 3 configurations of the wide tri-layered FFNN (the paper's best model)
- Best model: `[100, 200, 100]` with 2000 iterations
- Outputs: `wide_results_train`, `wide_results_test`, `best_net`,
  `best_pred_test`, `best_pred_real`, `best_true_real`

#### `s05_figures.m`
- Generates all 12 figures from the paper
- Saves each as a `.png` file in the working directory
- Requires outputs from all previous stages

#### `s06_tables.m`
- Prints all 5 tables to the MATLAB Command Window
- Saves each as a `.csv` file in the working directory
- Table 5 (method comparison) uses fixed values from the paper

---

## 6. How to Run

### Step 1 — Set your working directory
In MATLAB, navigate to the folder containing all the `.m` files:
```matlab
cd 'C:\path\to\your\project\folder'
```

### Step 2 — Confirm your dataset path
Open `s01_load_data.m` and check line 13:
```matlab
DATASET_ROOT = './Dataset_Li-ion';
```
If your dataset folder has a different name or location, update this path.
You can use an absolute path:
```matlab
DATASET_ROOT = 'C:\Users\YourName\Documents\Dataset_Li-ion';
```

### Step 3 — Run the master script
```matlab
run('main_soc_analysis.m')
```
Or simply open `main_soc_analysis.m` in the MATLAB Editor and press **Run (F5)**.

### Step 4 — Check outputs
After completion, your project folder will contain:
- 12 PNG figure files
- 5 CSV table files
- All results printed to the Command Window

### Expected Runtime
| Stage | Approximate Time |
|---|---|
| s01 — Load data (50k rows) | 30–60 seconds |
| s02 — Ensemble models | 1–3 minutes |
| s03 — 15 NN variants | 5–15 minutes |
| s04 — Wide tri-layer (3 configs) | 5–15 minutes |
| s05 — Figures (incl. convergence) | 5–10 minutes |
| s06 — Tables | < 1 second |
| **Total** | **~20–45 minutes** |

> To speed things up, reduce `MAX_ROWS` in `s01_load_data.m` (see Section 7).

---

## 7. Configuration & Tuning

All user-configurable settings are at the top of `s01_load_data.m`:

```matlab
DATASET_ROOT     = './Dataset_Li-ion';   % Path to dataset root
TARGET_SUBDIR    = '25degC';             % Temperature folder to use
MAX_ROWS         = 50000;                % Row cap (lower = faster)
CURRENT_TARGETS  = [0.75, 0.10];        % Target discharge currents (Amps)
CURRENT_BAND     = 0.15;                % ± tolerance around each target
```

### Reducing Runtime
| What to change | Effect |
|---|---|
| `MAX_ROWS = 10000` | Much faster, slightly lower accuracy |
| `MAX_ROWS = 20000` | Good balance of speed and accuracy |
| `MAX_ROWS = 50000` | Default — matches paper quality |

### Changing Current Filter
The `CURRENT_TARGETS` values match the discharge currents you specified (0.75A and 0.10A).
If no rows load, try widening the band:
```matlab
CURRENT_BAND = 0.30;   % wider filter — accepts more rows
```
Or check what current values actually exist in your files by running in MATLAB:
```matlab
% Quick diagnostic — run this BEFORE main_soc_analysis.m
tbl = readtable('Dataset_Li-ion/25degC/556_Charge1.csv', ...
    'NumHeaderLines', 3);
disp(unique(round(abs(tbl{:,4}), 1)))  % column 4 is usually Current
```

### Using a Different Temperature Folder
Change `TARGET_SUBDIR` in `s01_load_data.m`:
```matlab
TARGET_SUBDIR = '10degC';   % or '40degC', 'n10degC', 'n20degC'
```

---

## 8. Output Files

After running, these files are created in your project folder:

### PNG Figures (12 files)
```
Fig2_training_data_profile.png
Fig4_ensemble_training.png
Fig5_ensemble_testing.png
Fig10_all_nn_rmse.png
Fig11_residual_wide_nn.png
Fig12_residual_trilayer.png
Fig13_singlelayer_convergence.png
Fig14_singlelayer_regression.png
Fig15_trilayer_convergence.png
Fig16_trilayer_regression.png
Fig17_error_hist_singlelayer.png
Fig18_error_hist_trilayer.png
```

### CSV Tables (5 files)
```
Table1_ensemble_results.csv
Table2_nn_training.csv
Table3_nn_testing.csv
Table4_wide_trilayer.csv
Table5_method_comparison.csv
```

---

## 9. Figures Generated

| Figure | Description | Paper Figure |
|---|---|---|
| Fig 2 | 6-panel training data profile: Voltage, Current, Temperature, Avg V, Avg I, SOC | Fig. 2 |
| Fig 4 | MAE & RMSE scatter comparison — ensemble models (training data) | Fig. 4 |
| Fig 5 | MAE & RMSE scatter comparison — ensemble models (test data) | Fig. 5 |
| Fig 10 | Bar chart of RMSE across all 15 neural network model variants | Fig. 10 |
| Fig 11 | Residual plot for the wide single-layer neural network | Fig. 11 |
| Fig 12 | Residual plot for the wide tri-layered FFNN (best model) | Fig. 12 |
| Fig 13 | MSE convergence curve — single-layer FFNN vs iterations | Fig. 13 |
| Fig 14 | Predicted vs actual scatter — single-layer FFNN (train & test) | Fig. 14 |
| Fig 15 | MSE convergence curve — tri-layered FFNN vs iterations | Fig. 15 |
| Fig 16 | Predicted vs actual scatter — tri-layered FFNN (train & test) | Fig. 16 |
| Fig 17 | Error histogram with 20 bins — single-layer FFNN | Fig. 17 |
| Fig 18 | Error histogram with 20 bins — tri-layered FFNN | Fig. 18 |

---

## 10. Tables Generated

| Table | Description | Paper Table |
|---|---|---|
| Table 1 | MAE, MSE, RMSE, R² for Linear Regression, Tree, Bagged, Boosted — train & test | Table 1 |
| Table 2 | Training metrics for all 15 NN variants (5 architectures × 3 activations) | Table 2 |
| Table 3 | Testing metrics for all 15 NN variants (5 architectures × 3 activations) | Table 3 |
| Table 4 | Wide tri-layered FFNN: 3 configurations tested at 1000/1500/2000 iterations | Table 4 |
| Table 5 | Comparison of proposed FFNN with 12 related methods from literature | Table 5 |

> **Note on Table 5:** The comparison values (Kalman filters, LSTM, GRU-RNN, etc.)
> are taken directly from the paper and are fixed constants — they do not change
> when you re-run the code. Only the "Proposed FFNN (wide)" row uses your computed results.

---

## 11. Models Implemented

### Ensemble Models (s02)
| Model | MATLAB Function | Key Parameters |
|---|---|---|
| Linear Regression | `fitlm` | Default |
| Decision Tree | `fitrtree` | `MinLeafSize = 8` |
| Bagged Trees | `fitrensemble` | `Method='Bag'`, 30 learners, `MinLeafSize=8` |
| Boosted Trees | `fitrensemble` | `Method='LSBoost'`, 30 learners, `LearnRate=0.01` |

### Neural Networks (s03 + s04)
All neural networks use `fitrnet` from the Statistics and Machine Learning Toolbox.

| Model | Layer Sizes | Activations Tested | Iterations |
|---|---|---|---|
| Narrow FFNN | [10] | ReLU, Tanh, Sigmoid | 1000 |
| Medium FFNN | [25] | ReLU, Tanh, Sigmoid | 1000 |
| Wide FFNN | [100] | ReLU, Tanh, Sigmoid | 1000 |
| Bi-layered FFNN | [10, 10] | ReLU, Tanh, Sigmoid | 1000 |
| Tri-layered FFNN | [10, 10, 10] | ReLU, Tanh, Sigmoid | 1000 |
| Wide Tri-layer (a) | [100, 100, 100] | ReLU | 1500 |
| Wide Tri-layer (b) ⭐ | [100, 200, 100] | ReLU | 2000 |
| Wide Tri-layer (c) | [100, 100, 100] | ReLU | 1000 |

⭐ = Best performing model (paper's proposed architecture)

### Input Features
| Feature | Column in CSV | Description |
|---|---|---|
| Voltage | `Voltage` | Measured cell terminal voltage (V) |
| Current | `Current` | Measured current (A) |
| Temperature | `Temperature` | Battery case temperature (°C) |

### Target Variable
| Target | Column in CSV | Description |
|---|---|---|
| SOC | `SOC` | State of Charge (primary target) |
| Capacity | `Capacity` | Used as fallback if SOC column not found |

---

## 12. Known Issues & Fixes

### Issue: "Subfolder not found"
**Cause:** `DATASET_ROOT` path is wrong or MATLAB's working directory doesn't match.
**Fix:** Use an absolute path in `s01_load_data.m`:
```matlab
DATASET_ROOT = 'C:\Users\YourName\Documents\Dataset_Li-ion';
```

### Issue: "Arrays have incompatible sizes"
**Cause:** Some CSV files have rows with a different number of comma-separated fields
than the header row (ragged rows).
**Fix:** Already handled automatically — the parser pads short rows and trims long rows
to match the header column count exactly.

### Issue: "Column headers modified to valid MATLAB identifiers"
**Cause:** MATLAB renames columns with spaces (e.g. `Time Stamp`) by default.
**Fix:** Already handled — `'VariableNamingRule', 'preserve'` is set in all `readtable`
calls so original column names are kept intact.

### Issue: "DATETIME data was created using format MM/dd... but also matched dd/MM..."
**Cause:** MATLAB is unsure about the date format in the `Time Stamp` column.
**Fix:** Already handled — the `Time Stamp` column is explicitly assigned
`'InputFormat', 'MM/dd/uuuu hh:mm:ss aa'` via `setvaropts`.

### Issue: "No data loaded — no rows match current filter"
**Cause:** The 0.75A / 0.10A current filter finds no matching rows in your CSV files.
**Fix:** Widen the tolerance in `s01_load_data.m`:
```matlab
CURRENT_BAND = 0.30;
```
Or run this diagnostic to see what currents exist:
```matlab
f = dir('Dataset_Li-ion/25degC/*.csv');
tbl = parseSingleCSV(fullfile('Dataset_Li-ion/25degC', f(1).name));
disp(unique(round(abs(tbl.Current), 2)));
```

### Issue: `fitrnet` not found
**Cause:** Your MATLAB version is older than R2021b, or the Statistics and Machine
Learning Toolbox is not installed.
**Fix:** Run `ver('stats')` to check. If it's missing, contact your institution's
IT/software team to get the toolbox installed.

### Issue: Training is very slow
**Fix:** Reduce `MAX_ROWS` in `s01_load_data.m`:
```matlab
MAX_ROWS = 10000;   % fast — runs in ~5 minutes total
```

---

## 13. Troubleshooting

### Check your MATLAB working directory
```matlab
pwd                        % shows current directory
cd 'path/to/your/project'  % change to project folder
```

### Verify the dataset folder is visible
```matlab
dir('Dataset_Li-ion')
dir('Dataset_Li-ion/25degC')
```

### Check what columns are in your CSV
```matlab
% Run s01_load_data.m first, then:
rawData.Properties.VariableNames
```

### Re-run a single stage without re-running everything
Because all variables persist in the workspace, you can re-run any individual stage:
```matlab
run('s04_wide_trilayer.m')   % re-run just the best model stage
run('s05_figures.m')         % re-generate all figures
run('s06_tables.m')          % re-print all tables
```

### Clear and start fresh
```matlab
clc; clear; close all;
run('main_soc_analysis.m');
```

---

## Notes for School Submission

- The **values in Table 5** are from the published paper — your own results
  (Table 1–4) will differ slightly because you are using a subset of the dataset
  (25°C only, filtered current), whereas the paper used all temperatures and all
  discharge conditions. This is expected and should be noted in your report.

- Your results should still show the **same trend** as the paper: neural networks
  outperform ensemble methods, and the wider/deeper the network, the better the
  SOC prediction.

- The **best model** you should highlight in your report is the wide tri-layered
  FFNN from Stage 4 (`s04_wide_trilayer.m`), specifically the
  `[100, 200, 100]` configuration at 2000 iterations.

---

*Generated for school project use. Based on open-access paper by E.O. Ofoegbu, 2025.*