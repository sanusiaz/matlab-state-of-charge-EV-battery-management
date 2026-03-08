# SOC Estimation — LG HG2 Li-ion Battery
### Comparison of Ensemble Methods and Feedforward Neural Networks
### Mechatronics Engineering Final Year Project | Ahmadu Bello University, Zaria

---

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Requirements](#2-requirements)
3.  [Dataset Setup](#3-dataset-setup)
4.  [File Structure](#4-file-structure)
5.  [How to Run](#5-how-to-run)
6.  [Output Files](#6-output-files)
7.  [Configuration and Tuning](#7-configuration-and-tuning)
8.  [Models Implemented](#8-models-implemented)
9.  [Comparison Framework — s07](#9-comparison-framework--s07)
10. [Known Issues and Fixes Applied](#10-known-issues-and-fixes-applied)
11. [Troubleshooting](#11-troubleshooting)
12. [References](#12-references)

---

## 1. Project Overview

This project implements a **direct comparison** of ensemble methods and feedforward
neural networks (FFNNs) for State of Charge (SOC) estimation in a lithium-ion
electric vehicle battery. All 22 models are trained and evaluated on the same
dataset under identical conditions, and a dedicated comparison script
(s07_comparison.m) produces a full head-to-head ranking of both model families.

### What is SOC?
State of Charge is a measure of how much usable energy remains in a battery,
expressed as a value between 0 (empty) and 1 (fully charged):

    SOC(t) = Q_Remaining(t) / Q_Rated

SOC cannot be measured directly — it must be estimated from Voltage, Current,
and Temperature measurements. The Open Circuit Voltage (OCV) of the LG HG2 cell
is nearly flat between 20% and 93% SOC, making it impossible to read SOC from
voltage alone in that region. Machine learning models that learn from all three
inputs are therefore required.

### Battery Specifications
| Property              | Value                            |
|-----------------------|----------------------------------|
| Cell model            | LG 18650 HG2                     |
| Chemistry             | Li[NiMnCo]O2 / Graphite + SiO   |
| Nominal voltage       | 3.6 V                            |
| Rated capacity        | 3.0 Ah                           |
| Max charge voltage    | 4.2 V                            |
| Min discharge voltage | 2.5 V                            |
| Max discharge current | 20 A                             |
| Measurement accuracy  | 0.1% of full scale               |

### Models Compared
| Family              | Models Included                                           | Count |
|---------------------|-----------------------------------------------------------|-------|
| Ensemble Methods    | Linear Regression, Decision Tree, Bagged Trees,           |   4   |
|                     | Boosted Trees                                             |       |
| Standard FFNNs      | Narrow, Medium, Wide, Bi-layer, Tri-layer                 |  15   |
|                     | x 3 activations (ReLU, Tanh, Sigmoid)                     |       |
| Wide Tri-Layer FFNN | [100,100,100]x1500, [100,200,100]x2000,                   |   3   |
|                     | [100,100,100]x1000                                        |       |
| **Total**           |                                                           | **22**|

### Dataset
- **Source:** LG 18650HG2 Li-ion battery dataset (Kollmeyer, 2020)
- **Temperature:** 25 degrees C subset only
- **Discharge currents:** 0.10A, 0.75A, 1.5A, 2.0A, 3.0A (plus or minus 0.20A)
- **Max rows:** 500,000
- **Split:** 70% training / 30% testing, fixed seed rng(42)
- **Normalisation:** MinMax scaling to [0, 1]

---

## 2. Requirements

### MATLAB Toolboxes
| Toolbox                                 | Minimum Version | Functions Used                                       |
|-----------------------------------------|-----------------|------------------------------------------------------|
| Statistics and Machine Learning Toolbox | R2021b          | fitrnet, fitlm, fitrtree, fitrensemble, templateTree |

The Deep Learning Toolbox is **NOT required.**
All neural network training is handled by fitrnet() from the Statistics and
Machine Learning Toolbox.

### MATLAB Version
- Minimum required: R2021b
- Recommended: R2022b or R2023a

### Verify your toolbox
```matlab
ver('stats')
% Should show: Statistics and Machine Learning Toolbox  Version 12.x or higher
```

---

## 3. Dataset Setup

### Download
LG 18650HG2 Li-ion Battery Dataset — publicly available on Mendeley Data:
https://doi.org/10.17632/cp3473x7xv.3

Credit: Dr. Phillip Kollmeyer, McMaster University, Canada.

### Required Folder Structure
```
Dataset_Li-ion/
|-- 10degC/
|   +-- *.csv
|-- 25degC/          <-- ONLY THIS FOLDER IS READ
|   |-- 551_Charge1.csv
|   |-- 551_HWFET.csv
|   +-- ... (all CSV files for 25 degrees C)
|-- 40degC/
|-- n10degC/
+-- n20degC/
```

Only the 25degC subfolder is read. All other temperature folders are ignored.

---

## 4. File Structure

```
your_project_folder/
|
|-- main_soc_analysis.m     <- ENTRY POINT: run this to execute everything
|
|-- s01_load_data.m         <- Stage 1: Load, filter, normalise, split dataset
|-- s02_ensemble_models.m   <- Stage 2: 4 ensemble models (Table 1)
|-- s03_neural_networks.m   <- Stage 3: 15 standard FFNN configs (Tables 2-3)
|-- s04_wide_trilayer.m     <- Stage 4: 3 Wide Tri-Layer FFNN configs (Table 4)
|-- s05_figures.m           <- Stage 5: Figures 1-6 saved in figures/
|-- s06_tables.m            <- Stage 6: Tables 1-5 saved in tables/
|-- s07_comparison.m        <- Stage 7: Head-to-head comparison
|                                       Tables 6-7 in tables/
|                                       Figures 7-8 in figures/
|
|-- Dataset_Li-ion/         <- You provide this (download from Mendeley Data)
|   +-- 25degC/
|       +-- *.csv
|
|-- figures/                <- AUTO-CREATED by main_soc_analysis.m
|   +-- *.png               <- 15 figure files saved here
|
|-- tables/                 <- AUTO-CREATED by main_soc_analysis.m
|   +-- *.csv               <- 7 table files saved here
|
+-- README.md               <- This file
```

### Why figures/ and tables/ subfolders?
Keeping outputs in dedicated subfolders keeps the project root clean.
Both folders are created automatically by main_soc_analysis.m using mkdir()
before any stage runs — you do not need to create them manually.

### Detailed Role of Each Script

#### main_soc_analysis.m
Master runner. Creates figures/ and tables/ folders, then calls all 7 stage
scripts in order. Variables are shared through the MATLAB workspace. Do not
run clear between stages — earlier stage variables will be lost.

#### s01_load_data.m
- Reads all CSV files from Dataset_Li-ion/25degC/
- Drops non-essential columns per file before stacking (prevents type errors)
- Filters rows to plus or minus 0.20A bands around 5 target discharge current levels
- Applies MinMax normalisation to [0, 1] using training set parameters only
- Splits 70/30 with rng(42) fixed seed for reproducibility
- Key outputs: X_train, X_test, y_train, y_test, yMin, yMax

#### s02_ensemble_models.m
- Linear Regression (fitlm), Decision Tree (fitrtree, MinLeafSize=8)
- Bagged Trees (fitrensemble, Bag method, 30 learners)
- Boosted Trees (fitrensemble, LSBoost method, 30 learners, lr=0.01)
- Key outputs: results_train, results_test, ensemble_preds_test

#### s03_neural_networks.m
- 5 architectures x 3 activations = 15 FFNN configurations
- Architectures: Narrow[10], Medium[25], Wide[100], Bi-layer[10,10], Tri-layer[10,10,10]
- Activations: ReLU, Tanh, Sigmoid
- Key outputs: nn_results_train, nn_results_test, nn_labels

#### s04_wide_trilayer.m
- Config A: [100,100,100] — 1,500 iterations
- Config B: [100,200,100] — 2,000 iterations
- Config C: [100,100,100] — 1,000 iterations
- Key outputs: wide_results_train, wide_results_test, wide_labels, best_net

#### s05_figures.m
- Generates Figures 1-6 as PNG files into the figures/ folder
- Fig 6 (OCV vs SOC) uses a theoretical LiNMC polynomial model, not dataset data

#### s06_tables.m
- Generates Tables 1-5 as CSV files into the tables/ folder

#### s07_comparison.m
- Collects results from all 22 trained models
- Identifies the best model in each family by lowest test MAE
- Prints head-to-head summary table in the MATLAB Command Window
- Saves Table 6 (full ranking of all 22 models) to tables/
- Saves Table 7 (best ensemble vs best FFNN) to tables/
- Generates Figure 7 (MAE comparison bar chart) to figures/
- Generates Figure 8 (R-squared comparison chart) to figures/
- Declares the winning family and reports the percentage MAE difference

---

## 5. How to Run

### Step 1 — Set working directory in MATLAB
```matlab
cd 'C:\Users\YourName\Documents\SOC_Project'
```

### Step 2 — Confirm dataset path
Open s01_load_data.m and update DATASET_ROOT if needed:
```matlab
DATASET_ROOT = './Dataset_Li-ion';
% Or use a full absolute path:
DATASET_ROOT = 'C:\Users\YourName\Documents\Dataset_Li-ion';
```

### Step 3 — Run
```matlab
run('main_soc_analysis.m')
```
Or open main_soc_analysis.m in the MATLAB Editor and press F5.

### Step 4 — Monitor progress in Command Window
```
Output folders ready: figures/  tables/

--- Stage 1: Loading data ---
Found 25 CSV files in '25degC'
Final dataset: XXXXX rows  |  Target column: 'SOC'
Stage 1 complete.

--- Stage 2: Ensemble models ---
...

--- Stage 7: Head-to-Head Comparison (Ensemble vs FFNN) ---
RESULT: FFNN outperforms Ensemble by X.XX% on test MAE.
```

### Expected Runtime
| Stage | Description                             | Approximate Time |
|-------|-----------------------------------------|------------------|
| s01   | Load and filter up to 500,000 rows      | 2-5 minutes      |
| s02   | 4 ensemble models                       | 2-5 minutes      |
| s03   | 15 NN variants at 1000 iterations each  | 15-30 minutes    |
| s04   | 3 wide tri-layer configurations         | 10-20 minutes    |
| s05   | Figures 1-6                             | 5-15 minutes     |
| s06   | Tables 1-5                              | Under 1 second   |
| s07   | Comparison and Figures 7-8              | Under 2 minutes  |
| Total |                                         | 35-75 minutes    |

For a quick test: reduce MAX_ROWS to 20000 in s01_load_data.m (approx. 10-15 min total).

---

## 6. Output Files

### PNG Figures — saved in figures/ (15 files total)
| Filename                                     | Description                                       | Stage |
|----------------------------------------------|---------------------------------------------------|-------|
| Fig2_training_data_profile.png               | 6-panel feature profile of training data          | s05   |
| Fig4_ensemble_training.png                   | MAE and RMSE — ensemble models, training set      | s05   |
| Fig5_ensemble_testing.png                    | MAE and RMSE — ensemble models, test set          | s05   |
| Fig6_OCV_vs_SOC.png                          | OCV vs SOC theoretical reference curve            | s05   |
| Fig10_all_nn_rmse.png                        | RMSE bar chart — all 15 NN variants               | s05   |
| Fig11_residual_wide_nn.png                   | Residual plot — wide single-layer NN              | s05   |
| Fig12_residual_trilayer.png                  | Residual plot — wide tri-layered FFNN             | s05   |
| Fig13_singlelayer_convergence.png            | MSE convergence — single-layer FFNN               | s05   |
| Fig14_singlelayer_regression.png             | Predicted vs actual — single-layer FFNN           | s05   |
| Fig15_trilayer_convergence.png               | MSE convergence — tri-layered FFNN                | s05   |
| Fig16_trilayer_regression.png                | Predicted vs actual — tri-layered FFNN            | s05   |
| Fig17_error_hist_singlelayer.png             | Error histogram — single-layer FFNN               | s05   |
| Fig18_error_hist_trilayer.png                | Error histogram — tri-layered FFNN                | s05   |
| Figure7_ensemble_vs_ffnn_comparison.png      | Head-to-head MAE comparison bar chart             | s07   |
| Figure8_r2_comparison.png                    | Head-to-head R-squared comparison chart           | s07   |

### CSV Tables — saved in tables/ (7 files total)
| Filename                            | Contents                                        | Stage |
|-------------------------------------|-------------------------------------------------|-------|
| Table1_ensemble_results.csv         | Ensemble — MAE, MSE, RMSE, R2 (train and test)  | s06   |
| Table2_nn_training.csv              | 15 FFNN configs — training metrics              | s06   |
| Table3_nn_testing.csv               | 15 FFNN configs — test metrics                  | s06   |
| Table4_wide_trilayer.csv            | Wide Tri-Layer 3 configs — train and test       | s06   |
| Table5_method_comparison.csv        | Literature comparison with published methods    | s06   |
| Table6_full_comparison_ranking.csv  | All 22 models ranked by test MAE                | s07   |
| Table7_head_to_head.csv             | Best Ensemble vs Best FFNN — all metrics        | s07   |

---

## 7. Configuration and Tuning

All key settings are at the top of s01_load_data.m:

```matlab
DATASET_ROOT     = './Dataset_Li-ion';             % path to dataset root
TARGET_SUBDIR    = '25degC';                       % temperature subfolder
MAX_ROWS         = 500000;                         % row cap
CURRENT_TARGETS  = [0.75, 0.10, 1.5, 2.0, 3.0];  % target discharge currents (A)
CURRENT_BAND     = 0.20;                           % +/- tolerance (A)
```

### Current Filter Acceptance Bands
| Target (A) | Lower (A) | Upper (A) |
|------------|-----------|-----------|
| 0.10       | 0.00      | 0.30      |
| 0.75       | 0.55      | 0.95      |
| 1.50       | 1.30      | 1.70      |
| 2.00       | 1.80      | 2.20      |
| 3.00       | 2.80      | 3.20      |

### Speed vs Accuracy
| MAX_ROWS | Approx. Runtime | Quality          |
|----------|----------------|------------------|
| 10,000   | 5-10 min       | Low — quick test |
| 20,000   | 10-15 min      | Reasonable       |
| 50,000   | 15-25 min      | Good             |
| 200,000  | 25-45 min      | High             |
| 500,000  | 35-75 min      | Best achievable  |

---

## 8. Models Implemented

### Ensemble Models — s02_ensemble_models.m
| Model             | MATLAB Function | Key Parameters                          |
|-------------------|-----------------|-----------------------------------------|
| Linear Regression | fitlm           | Default                                 |
| Decision Tree     | fitrtree        | MinLeafSize = 8                         |
| Bagged Trees      | fitrensemble    | Method=Bag, 30 cycles, MinLeafSize=8    |
| Boosted Trees     | fitrensemble    | Method=LSBoost, 30 cycles, lr=0.01      |

### Standard FFNNs — s03_neural_networks.m
| Architecture | Layers       | Activations Tested  | Iterations |
|--------------|--------------|---------------------|------------|
| Narrow       | [10]         | ReLU, Tanh, Sigmoid | 1,000      |
| Medium       | [25]         | ReLU, Tanh, Sigmoid | 1,000      |
| Wide         | [100]        | ReLU, Tanh, Sigmoid | 1,000      |
| Bi-layered   | [10, 10]     | ReLU, Tanh, Sigmoid | 1,000      |
| Tri-layered  | [10, 10, 10] | ReLU, Tanh, Sigmoid | 1,000      |

### Wide Tri-Layer FFNNs — s04_wide_trilayer.m
| Config | Layers          | Iterations |
|--------|-----------------|------------|
| A      | [100, 100, 100] | 1,500      |
| B      | [100, 200, 100] | 2,000      |
| C      | [100, 100, 100] | 1,000      |

All neural networks use fitrnet() from the Statistics and Machine Learning Toolbox
with L-BFGS optimisation. No Deep Learning Toolbox is required.

### Features and Target
| Role    | Column      | Description                     |
|---------|-------------|---------------------------------|
| Feature | Voltage     | Cell terminal voltage (V)       |
| Feature | Current     | Discharge current (A)           |
| Feature | Temperature | Cell surface temperature (C)    |
| Target  | SOC         | State of Charge — value 0 to 1  |

### Preprocessing Pipeline
1. Drop non-essential columns per CSV before stacking
2. Filter rows to accepted plus or minus 0.20A current bands
3. Remove rows with any NaN in features or target
4. MinMax scaling: x_norm = (x - x_min) / (x_max - x_min)
5. Shuffle with rng(42) then split 70% training / 30% testing
6. Store yMin and yMax for inverse-transform in plotting

---

## 9. Comparison Framework — s07

s07_comparison.m is the dedicated script that makes this project a comparison study.
It runs after all 22 models have been trained.

### What it does
1. Finds the best model in each family by lowest test MAE
2. Prints a head-to-head comparison table in the MATLAB Command Window
3. Ranks all 22 models together from best to worst
4. Declares a winner and calculates the percentage MAE improvement
5. Saves Tables 6 and 7 to the tables/ folder
6. Saves Figures 7 and 8 to the figures/ folder

### Sample Command Window Output
```
=================================================================
  HEAD-TO-HEAD COMPARISON: Best Ensemble vs Best FFNN
  (Test Set Results)
=================================================================
Family     Best Model                      MAE          R2
-----------------------------------------------------------------
Ensemble   Bagged Trees                    X.XXXXXXX    X.XXXX
FFNN       ReLU [100:200:100] 2000 iter    X.XXXXXXX    X.XXXX
-----------------------------------------------------------------

  RESULT: FFNN outperforms Ensemble by X.XX% on test MAE.
```

---

## 10. Known Issues and Fixes Applied

All issues below are already fixed in the code. No action is required.

| Error                                        | Root Cause                                   | Fix Applied                                        |
|----------------------------------------------|----------------------------------------------|----------------------------------------------------|
| Error concatenating Prog Time using VERTCAT  | Prog Time stored as different types          | Non-essential columns dropped per file before stack|
| Arrays have incompatible sizes               | Ragged CSV rows                              | Parser pads rows to match header column count      |
| Column headers modified to valid MATLAB IDs  | MATLAB auto-renames columns containing spaces| VariableNamingRule set to preserve                 |
| DATETIME matched both MM/dd and dd/MM        | Ambiguous date format in CSV files           | setvaropts pins format to MM/dd/uuuu hh:mm:ss aa  |
| Function definitions must appear at end      | computeMetrics placed before executable code | Moved to bottom of s02_ensemble_models.m           |
| All table variables must have same row count | wide_labels had 3 entries but 6 rows needed  | wide_labels doubled with [wide_labels;wide_labels] |

---

## 11. Troubleshooting

### Dataset folder not found
```matlab
pwd
isfolder('./Dataset_Li-ion')           % should return 1
isfolder('./Dataset_Li-ion/25degC')    % should return 1
dir('./Dataset_Li-ion/25degC')         % should list CSV files
```
If any return 0, update DATASET_ROOT in s01_load_data.m to the full absolute path.

### Stage 1 reports zero rows loaded
```matlab
% Widen the tolerance in s01_load_data.m:
CURRENT_BAND = 0.40;
```
Or check what currents exist in your data:
```matlab
csvFiles = dir('Dataset_Li-ion/25degC/*.csv');
f = fullfile('Dataset_Li-ion/25degC', csvFiles(1).name);
t = readtable(f, 'NumHeaderLines', 3);
disp(unique(round(abs(t{:,4}), 2)))
```

### fitrnet not recognised
```matlab
ver('stats')    % needs Version 12.0 (R2021b) or later
```
Contact your institution IT team if the toolbox is missing.

### figures/ or tables/ folder missing
```matlab
if ~exist('figures','dir'), mkdir('figures'); end
if ~exist('tables','dir'),  mkdir('tables');  end
```

### Re-run a single stage without restarting
```matlab
run('s07_comparison.m')    % re-run comparison only
run('s05_figures.m')        % re-generate all figures
run('s06_tables.m')         % re-save all tables
```

### Start completely fresh
```matlab
clc; clear; close all;
run('main_soc_analysis.m');
```

---

## 12. References

1. Kollmeyer, P. (2020).
   LG 18650HG2 Li-ion Battery Data (Version 3).
   Mendeley Data.
   https://doi.org/10.17632/cp3473x7xv.3

2. Breiman, L. (1996).
   Bagging predictors.
   Machine Learning, 24(2), 123-140.
   https://doi.org/10.1007/BF00058655

3. Friedman, J.H. (2001).
   Greedy function approximation: a gradient boosting machine.
   Annals of Statistics, 29(5), 1189-1232.
   https://doi.org/10.1214/aos/1013203451

4. Cybenko, G. (1989).
   Approximation by superpositions of a sigmoidal function.
   Mathematics of Control, Signals and Systems, 2(4), 303-314.
   https://doi.org/10.1007/BF02551274

5. Hornik, K., Stinchcombe, M., and White, H. (1989).
   Multilayer feedforward networks are universal approximators.
   Neural Networks, 2(5), 359-366.
   https://doi.org/10.1016/0893-6080(89)90020-8

6. MathWorks. (2021).
   fitrnet — Train neural network for regression.
   MATLAB Statistics and Machine Learning Toolbox Documentation.
   https://www.mathworks.com/help/stats/fitrnet.html