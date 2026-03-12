# SOC Estimation — LG HG2 Li-ion Battery
### Comparison of Ensemble Methods and Feedforward Neural Networks
### Automotive Engineering Final Year Project | Ahmadu Bello University, Zaria

---

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Preliminary Results Summary](#2-preliminary-results-summary)
3.  [Requirements](#3-requirements)
4.  [Dataset Setup](#4-dataset-setup)
5.  [File Structure](#5-file-structure)
6.  [How to Run](#6-how-to-run)
7.  [Output Files](#7-output-files)
8.  [Configuration and Tuning](#8-configuration-and-tuning)
9.  [Models Implemented](#9-models-implemented)
10. [Comparison Framework — s07](#10-comparison-framework--s07)
11. [Known Issues and Fixes Applied](#11-known-issues-and-fixes-applied)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

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
| Property              | Value                          |
|-----------------------|--------------------------------|
| Cell model            | LG 18650 HG2                   |
| Chemistry             | Li[NiMnCo]O2 / Graphite + SiO |
| Nominal voltage       | 3.6 V                          |
| Rated capacity        | 3.0 Ah                         |
| Max charge voltage    | 4.2 V                          |
| Min discharge voltage | 2.5 V                          |
| Max discharge current | 20 A                           |
| Measurement accuracy  | 0.1% of full scale             |

### Models Compared
| Family              | Models Included                                       | Count |
|---------------------|-------------------------------------------------------|-------|
| Ensemble Methods    | Linear Regression, Decision Tree, Bagged Trees,       |   4   |
|                     | Boosted Trees                                         |       |
| Standard FFNNs      | Narrow, Medium, Wide, Bi-layer, Tri-layer             |  15   |
|                     | x 3 activations (ReLU, Tanh, Sigmoid)                 |       |
| Wide Tri-Layer FFNN | [100,100,100]x1500, [100,200,100]x2000,               |   3   |
|                     | [100,100,100]x1000                                    |       |
| **Total**           |                                                       | **22**|

### Dataset
- **Source:** LG 18650HG2 Li-ion battery dataset (Kollmeyer, 2020)
- **Temperature:** 25 degrees C subset only
- **Discharge currents:** 0.10A, 0.75A, 1.5A, 2.0A, 3.0A (plus or minus 0.20A)
- **Max rows:** 500,000
- **Split:** 70% training / 30% testing, fixed seed rng(42)
- **Normalisation:** MinMax scaling to [0, 1]

---

## 2. Preliminary Results Summary

These results were produced by running the full pipeline on the 25 degrees C
subset of the LG 18650HG2 dataset. All values are on the test set (30% held-out).

### Head-to-Head: Best Ensemble vs Best FFNN
| Family   | Best Model                    | Test MAE  | Test MSE  | Test RMSE | Test R²  |
|----------|-------------------------------|-----------|-----------|-----------|----------|
| Ensemble | Decision Tree                 | 0.0055553 | 0.0004667 | 0.0216027 | 0.9808   |
| FFNN     | ReLU [100:100:100] 1500 iter  | 0.0120298 | 0.0012038 | 0.0346963 | 0.9504   |

**Winner: Ensemble (Decision Tree) — outperformed best FFNN by 116.55% on test MAE**

### Ensemble Models — Test Set Results
| Model             | Test MAE  | Test RMSE | Test R²  |
|-------------------|-----------|-----------|----------|
| Decision Tree     | 0.0055553 | 0.0216027 | 0.9808   |
| Bagged Trees      | 0.0175466 | 0.0330710 | 0.9550   |
| Linear Regression | 0.0216166 | 0.0448073 | 0.9173   |
| Boosted Trees     | 0.1932300 | 0.2264609 | -1.1121  |

Note: Boosted Trees (LSBoost) performed poorly on this dataset — the negative R²
indicates the model underfit significantly. This is consistent with known limitations
of LSBoost at low learning rates on small normalised datasets.

### Best FFNN per Architecture — Test Set
| Architecture          | Best Activation | Test MAE  | Test R²  |
|-----------------------|-----------------|-----------|----------|
| Tri-layer [10,10,10]  | ReLU            | 0.0121276 | 0.9429   |
| Narrow [10]           | ReLU            | 0.0122420 | 0.9410   |
| Medium [25]           | ReLU            | 0.0124538 | 0.9384   |
| Bi-layer [10,10]      | Tanh            | 0.0125349 | 0.9446   |
| Wide [100]            | ReLU            | 0.0127206 | 0.9501   |

### Wide Tri-Layer FFNN — Test Set Results
| Configuration             | Test MAE  | Test RMSE | Test R²  |
|---------------------------|-----------|-----------|----------|
| ReLU [100:100:100] 1500   | 0.0120298 | 0.0346963 | 0.9504   |
| ReLU [100:100:100] 1000   | 0.0124316 | 0.0357275 | 0.9474   |
| ReLU [100:200:100] 2000   | 0.0126661 | 0.0335550 | 0.9536   |

### Full Ranking — All 22 Models (Test MAE, low to high)
| Rank | Family      | Model                        | Test MAE  | Test R²  |
|------|-------------|------------------------------|-----------|----------|
| 1    | Ensemble    | Decision Tree                | 0.0055553 | 0.9808   |
| 2    | FFNN-Wide   | ReLU [100:100:100] 1500 iter | 0.0120298 | 0.9504   |
| 3    | FFNN        | Tri-layer (ReLU)             | 0.0121276 | 0.9429   |
| 4    | FFNN        | Narrow (ReLU)                | 0.0122420 | 0.9410   |
| 5    | FFNN-Wide   | ReLU [100:100:100] 1000 iter | 0.0124316 | 0.9474   |
| 6    | FFNN        | Medium (ReLU)                | 0.0124538 | 0.9384   |
| 7    | FFNN        | Bi-layer (Tanh)              | 0.0125349 | 0.9446   |
| 8    | FFNN-Wide   | ReLU [100:200:100] 2000 iter | 0.0126661 | 0.9536   |
| 9    | FFNN        | Wide (ReLU)                  | 0.0127206 | 0.9501   |
| 10   | FFNN        | Bi-layer (Sigmoid)           | 0.0128888 | 0.9398   |
| 11   | FFNN        | Tri-layer (Sigmoid)          | 0.0129659 | 0.9409   |
| 12   | FFNN        | Bi-layer (ReLU)              | 0.0131897 | 0.9507   |
| 13   | FFNN        | Tri-layer (Tanh)             | 0.0137390 | 0.9441   |
| 14   | FFNN        | Medium (Tanh)                | 0.0138162 | 0.9401   |
| 15   | FFNN        | Narrow (Sigmoid)             | 0.0141590 | 0.9394   |
| 16   | FFNN        | Narrow (Tanh)                | 0.0142374 | 0.9397   |
| 17   | FFNN        | Wide (Tanh)                  | 0.0144950 | 0.9395   |
| 18   | FFNN        | Wide (Sigmoid)               | 0.0146104 | 0.9383   |
| 19   | FFNN        | Medium (Sigmoid)             | 0.0148310 | 0.9385   |
| 20   | Ensemble    | Bagged Trees                 | 0.0175466 | 0.9550   |
| 21   | Ensemble    | Linear Regression            | 0.0216166 | 0.9173   |
| 22   | Ensemble    | Boosted Trees                | 0.1932300 | -1.1121  |

### Key Observations
- ReLU activation consistently outperformed Tanh and Sigmoid across most architectures
- Wider networks did not always give lower error — the narrow Tri-layer (ReLU) ranked
  higher than the Wide single-layer (ReLU), suggesting depth matters more than width
  for this dataset at these iteration counts
- The Decision Tree (a single model) outperformed all 18 FFNN configurations on test MAE,
  which is a significant and unexpected finding worth discussing in the final report
- Boosted Trees is an outlier — its negative R² suggests it failed to converge properly
  under the LSBoost settings used and should be discussed as a limitation

---

## 3. Requirements

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

## 4. Dataset Setup

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

## 5. File Structure

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

### Detailed Role of Each Script

#### main_soc_analysis.m
Master runner. Creates figures/ and tables/ folders, then calls all 7 stage
scripts in order. Variables are shared through the MATLAB workspace. Do not
run clear between stages — earlier stage variables will be lost.

#### s01_load_data.m
- Reads all CSV files from Dataset_Li-ion/25degC/
- Drops non-essential columns per file before stacking
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
- Key outputs: nn_results_train, nn_results_test, nn_labels, wide_relu_pred_test

#### s04_wide_trilayer.m
- Config A: [100,100,100] — 1,500 iterations
- Config B: [100,200,100] — 2,000 iterations
- Config C: [100,100,100] — 1,000 iterations
- Key outputs: wide_results_train, wide_results_test, wide_labels, best_net

#### s05_figures.m
- Generates all 13 figures as PNG files into the figures/ folder
- Includes memory management block to prevent out-of-memory errors
- Fig 6 (OCV vs SOC) uses a theoretical LiNMC polynomial model, not dataset data

#### s06_tables.m
- Generates Tables 1-5 as CSV files into the tables/ folder
- Table 5 contains literature comparison values

#### s07_comparison.m
- Collects results from all 22 trained models
- Identifies the best model in each family by lowest test MAE
- Prints head-to-head summary table in the MATLAB Command Window
- Saves Tables 6 and 7 to tables/ folder
- Generates Figures 7 and 8 to figures/ folder
- Declares the winning family and reports the percentage MAE difference

---

## 6. How to Run

### Step 1 — Set working directory in MATLAB
```matlab
cd 'C:\Users\YourName\Documents\SOC_Project'
```

### Step 2 — Confirm dataset path
Open s01_load_data.m and update DATASET_ROOT if needed:
```matlab
DATASET_ROOT = './Dataset_Li-ion';
```

### Step 3 — Run
```matlab
run('main_soc_analysis.m')
```

### Expected Runtime
| Stage | Description                            | Approximate Time |
|-------|----------------------------------------|------------------|
| s01   | Load and filter up to 500,000 rows     | 2-5 minutes      |
| s02   | 4 ensemble models                      | 2-5 minutes      |
| s03   | 15 NN variants at 1000 iterations each | 15-30 minutes    |
| s04   | 3 wide tri-layer configurations        | 10-20 minutes    |
| s05   | Figures 1-6                            | 5-15 minutes     |
| s06   | Tables 1-5                             | Under 1 second   |
| s07   | Comparison and Figures 7-8             | Under 2 minutes  |
| Total |                                        | 35-75 minutes    |

For a quick test: reduce MAX_ROWS to 20000 in s01_load_data.m (approx. 10-15 min).

---

## 7. Output Files

### PNG Figures — saved in figures/ (15 files total)
| Filename                                | Description                                  | Stage |
|-----------------------------------------|----------------------------------------------|-------|
| Fig2_training_data_profile.png          | 6-panel feature profile of training data     | s05   |
| Fig4_ensemble_training.png              | MAE and RMSE — ensemble models, training set | s05   |
| Fig5_ensemble_testing.png               | MAE and RMSE — ensemble models, test set     | s05   |
| Fig6_OCV_vs_SOC.png                     | OCV vs SOC theoretical reference curve       | s05   |
| Fig10_all_nn_rmse.png                   | RMSE bar chart — all 15 NN variants          | s05   |
| Fig11_residual_wide_nn.png              | Residual plot — wide single-layer NN         | s05   |
| Fig12_residual_trilayer.png             | Residual plot — wide tri-layered FFNN        | s05   |
| Fig13_singlelayer_convergence.png       | MSE convergence — single-layer FFNN          | s05   |
| Fig14_singlelayer_regression.png        | Predicted vs actual — single-layer FFNN      | s05   |
| Fig15_trilayer_convergence.png          | MSE convergence — tri-layered FFNN           | s05   |
| Fig16_trilayer_regression.png           | Predicted vs actual — tri-layered FFNN       | s05   |
| Fig17_error_hist_singlelayer.png        | Error histogram — single-layer FFNN          | s05   |
| Fig18_error_hist_trilayer.png           | Error histogram — tri-layered FFNN           | s05   |
| Figure7_ensemble_vs_ffnn_comparison.png | Head-to-head MAE comparison bar chart        | s07   |
| Figure8_r2_comparison.png               | Head-to-head R-squared comparison chart      | s07   |

### CSV Tables — saved in tables/ (7 files total)
| Filename                           | Contents                                     | Stage |
|------------------------------------|----------------------------------------------|-------|
| Table1_ensemble_results.csv        | Ensemble — MAE, MSE, RMSE, R2 (train + test) | s06   |
| Table2_nn_training.csv             | 15 FFNN configs — training metrics           | s06   |
| Table3_nn_testing.csv              | 15 FFNN configs — test metrics               | s06   |
| Table4_wide_trilayer.csv           | Wide Tri-Layer 3 configs — train + test      | s06   |
| Table5_method_comparison.csv       | Literature comparison with published methods | s06   |
| Table6_full_comparison_ranking.csv | All 22 models ranked by test MAE             | s07   |
| Table7_head_to_head.csv            | Best Ensemble vs Best FFNN — all metrics     | s07   |

---

## 8. Configuration and Tuning

All key settings are at the top of s01_load_data.m:

```matlab
DATASET_ROOT     = './Dataset_Li-ion';
TARGET_SUBDIR    = '25degC';
MAX_ROWS         = 500000;
CURRENT_TARGETS  = [0.75, 0.10, 1.5, 2.0, 3.0];
CURRENT_BAND     = 0.20;
```

### Current Filter Acceptance Bands
| Target (A) | Lower (A) | Upper (A) |
|------------|-----------|-----------|
| 0.10       | 0.00      | 0.30      |
| 0.75       | 0.55      | 0.95      |
| 1.50       | 1.30      | 1.70      |
| 2.00       | 1.80      | 2.20      |
| 3.00       | 2.80      | 3.20      |

---

## 9. Models Implemented

### Ensemble Models — s02_ensemble_models.m
| Model             | MATLAB Function | Key Parameters                       |
|-------------------|-----------------|--------------------------------------|
| Linear Regression | fitlm           | Default                              |
| Decision Tree     | fitrtree        | MinLeafSize = 8                      |
| Bagged Trees      | fitrensemble    | Method=Bag, 30 cycles, MinLeafSize=8 |
| Boosted Trees     | fitrensemble    | Method=LSBoost, 30 cycles, lr=0.01   |

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

---

## 10. Comparison Framework — s07

s07_comparison.m is the dedicated script that makes this a comparison study.
It runs after all 22 models have been trained and produces the core findings.

### What it does
1. Finds the best model in each family by lowest test MAE
2. Prints a head-to-head comparison table in the MATLAB Command Window
3. Ranks all 22 models together from best to worst
4. Declares the winner and calculates the percentage MAE difference
5. Saves Tables 6 and 7 to the tables/ folder
6. Saves Figures 7 and 8 to the figures/ folder

---

## 11. Known Issues and Fixes Applied

| Error                                       | Root Cause                                   | Fix Applied                                        |
|---------------------------------------------|----------------------------------------------|----------------------------------------------------|
| Error concatenating Prog Time using VERTCAT | Prog Time stored as different types          | Non-essential columns dropped per file before stack|
| Arrays have incompatible sizes              | Ragged CSV rows                              | Parser pads rows to match header column count      |
| Column headers modified to valid MATLAB IDs | MATLAB auto-renames columns containing spaces| VariableNamingRule set to preserve                 |
| DATETIME matched both MM/dd and dd/MM       | Ambiguous date format in CSV files           | setvaropts pins format to MM/dd/uuuu hh:mm:ss aa  |
| Function definitions must appear at end     | computeMetrics placed before executable code | Moved to bottom of s02_ensemble_models.m           |
| Out of memory during Stage 5               | Large model objects still in workspace       | Memory management block added at top of s05        |
| pack has been removed                       | pack not supported in MATLAB R2023a+         | pack line removed, clear used instead              |

---

## 12. Troubleshooting

### Dataset folder not found
```matlab
pwd
isfolder('./Dataset_Li-ion')
isfolder('./Dataset_Li-ion/25degC')
dir('./Dataset_Li-ion/25degC')
```

### Stage 1 reports zero rows loaded
```matlab
CURRENT_BAND = 0.40;   % widen tolerance in s01_load_data.m
```

### fitrnet not recognised
```matlab
ver('stats')    % needs Version 12.0 (R2021b) or later
```

### Re-run a single stage without restarting
```matlab
run('s07_comparison.m')
run('s05_figures.m')
run('s06_tables.m')
```

### Start completely fresh
```matlab
clc; clear; close all;
run('main_soc_analysis.m');
```

---

## 13. References

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