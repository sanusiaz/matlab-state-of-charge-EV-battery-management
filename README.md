# SOC Estimation — LG HG2 Li-ion Battery
### School Project | Methodology Reference: Ofoegbu, *Journal of Energy Storage*, 2025

---

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Academic Integrity Statement](#2-academic-integrity-statement)
3.  [Paper Reference](#3-paper-reference)
4.  [How This Project Differs From the Paper](#4-how-this-project-differs-from-the-paper)
5.  [Requirements](#5-requirements)
6.  [Dataset Setup](#6-dataset-setup)
7.  [File Structure](#7-file-structure)
8.  [How to Run](#8-how-to-run)
9.  [Configuration and Tuning](#9-configuration-and-tuning)
10. [Output Files](#10-output-files)
11. [Figures Generated](#11-figures-generated)
12. [Tables Generated](#12-tables-generated)
13. [Models Implemented](#13-models-implemented)
14. [Known Issues and Fixes Applied](#14-known-issues-and-fixes-applied)
15. [Troubleshooting](#15-troubleshooting)
16. [Notes for Report Writing](#16-notes-for-report-writing)
17. [References](#17-references)

---

## 1. Project Overview

This project implements a data-driven State of Charge (SOC) estimation system for a
Lithium-ion (Li-ion) battery using MATLAB. It applies the methodology described in the
reference paper (Ofoegbu, 2025) to a different, independently chosen subset of the
LG 18650HG2 battery dataset, producing original results that can be compared against
the paper's findings.

### What is SOC?
State of Charge is a measure of how much usable energy remains in a battery,
expressed as a percentage between 0% (empty) and 100% (fully charged).
It is defined as:

    SOC(t) = Q_Remaining(t) / Q_Rated x 100%

Accurate SOC estimation is critical in Battery Management Systems (BMS) for electric
vehicles. Overcharging or deep discharging a Li-ion cell causes permanent damage
and potential safety hazards.

### Why is SOC Estimation Difficult?
The Open Circuit Voltage (OCV) of a Li-ion cell is nearly flat between 20% and 93% SOC
(see Fig 6 output by this project). This means you cannot reliably infer SOC from
voltage alone under normal operating conditions. A model that learns from Voltage,
Current, and Temperature measurements is therefore required.

### Approach
Three categories of models are trained and compared:
- Linear Regression          — simple baseline
- Ensemble Methods           — Decision Tree, Bagged Trees, Boosted Trees
- Feedforward Neural Networks (FFNN) — Narrow, Medium, Wide, Bi-layered, Tri-layered

The best performing model is a Wide Tri-layered FFNN with architecture [100, 200, 100]
trained using ReLU activation for 2000 iterations.

### Battery Specifications
| Property              | Value                            |
|-----------------------|----------------------------------|
| Cell model            | LG 18650 HG2                     |
| Chemistry             | Li[NiMnCo]O2 / Graphite + SiO   |
| Nominal voltage       | 3.6 V                            |
| Rated capacity        | 3.0 Ah                           |
| Max charge voltage    | 4.2 V                            |
| Min discharge voltage | 2.5 V                            |
| Max charge current    | 4 A                              |
| Max discharge current | 20 A                             |

---

## 2. Academic Integrity Statement

This section is important. Please read it carefully before submitting your project.

This project uses the methodology from Ofoegbu (2025) as a reference framework only.

### What was taken from the paper
- The general model architecture concept (FFNN with ReLU, tri-layered structure)
- The choice of input features (Voltage, Current, Temperature)
- The model comparison framework (ensemble vs neural network)
- Reference values in Table 5 (Kalman filter, LSTM, GRU comparisons from literature)
- Fig 6 (OCV vs SOC curve, reproduced from Berglund et al., 2019, cited as ref [32])

### What is independently generated in this project
- All numerical results in Tables 1 to 4 are computed by running the MATLAB
  code on a different data subset — they differ from the paper's values
- All figures (Fig 2, 4, 5, 6, 10 to 18) are generated from your own run
- The dataset subset is different from what the paper used (see Section 4)

### Is this academically acceptable?
Yes — provided you properly cite the paper and are transparent about what
you reproduced. Using a published paper's methodology and applying it to a
subset of the same dataset is standard academic practice.

What would NOT be acceptable:
- Copying the paper's numerical results without running the code yourself
- Presenting the paper's figures as your own without attribution
- Claiming you invented the methodology
- Manually editing your table numbers to match or differ from the paper

As long as you cite Ofoegbu (2025) clearly and explain that your results come
from your own independent run on a specific subset of the data, this project
is fully academically sound.

### Recommended citation phrasing for your report
    "This study adopts the methodology proposed by Ofoegbu (2025), applying it
    to a subset of the LG 18650HG2 dataset recorded at 25 degrees C with
    discharge currents of approximately 0.10A, 0.75A, 1.5A, 2.0A, and 3.0A.
    All reported results were independently computed and differ from those in
    the original paper due to the different data conditions used."

---

## 3. Paper Reference

Ofoegbu, E.O. (2025).
State of charge (SOC) estimation in electric vehicle (EV) battery management
systems using ensemble methods and neural networks.
Journal of Energy Storage, 114, 115833.
DOI: https://doi.org/10.1016/j.est.2025.115833

### Key results from the paper (for comparison reference only)
| Model                          | MAE (%) | MSE     |
|--------------------------------|---------|---------|
| Wide Tri-layered FFNN (paper)  | 0.88    | ~1e-08  |
| DNN                            | 1.10    | —       |
| Load-classifying NN            | 3.80    | —       |
| GRU-RNN                        | —       | 1e-06   |
| LSTM                           | 2.36    | —       |
| Kalman filter (UKF)            | 1.05    | —       |
| Kalman filter (CDKF)           | 0.96    | —       |
| Adaptive particle filtering    | 4.62    | —       |

Do NOT present these values as your own results.
They are from the paper and must be cited accordingly.

---

## 4. How This Project Differs From the Paper

This is the most important section for your academic justification.

| Aspect                   | Paper (Ofoegbu, 2025)                         | This Project                                        |
|--------------------------|-----------------------------------------------|-----------------------------------------------------|
| Temperature conditions   | All 5: 10C, 25C, 40C, -10C, -20C             | 25C only                                            |
| Discharge conditions     | All drive cycles: UDDS, HWFET, LA92, US06     | 0.10A, 0.75A, 1.5A, 2.0A, 3.0A (+/-0.20A)          |
| Total records used       | ~835,248 rows                                 | Up to 500,000 (actual count depends on 25C folder)  |
| Train / test split       | 70% / 30%                                     | 70% / 30% (same)                                    |
| Scaling method           | MinMax [0, 1]                                 | MinMax [0, 1] (same)                                |
| MATLAB toolbox           | Deep Learning Toolbox                         | Statistics and ML Toolbox only                      |
| Neural network function  | trainnet / patternnet                         | fitrnet                                             |
| Optimizer                | Adam                                          | L-BFGS (fitrnet default)                            |
| Tables 1 to 4 values     | Paper's computed results                      | Your independently computed results                 |
| Table 5 values           | Paper's literature comparison                 | Fixed from paper — cited as reference               |

### Note on row count
The 25C subfolder may not contain 500,000 rows matching the five current targets
even when combined. The code loads all it can find up to the cap. If Stage 1
reports a lower number such as 87,000 that is normal — it simply means the
25C folder does not have more matching data available.
The actual final count is printed at the end of Stage 1:

    Final dataset: XXXXX rows  |  Target column: 'SOC'

If your supervisor requires more data, the options are:
- Remove the current filter to use all discharge currents at 25C
- Include additional temperature folders such as 10C or 40C
Both options require only small changes in s01_load_data.m (see Section 9).

### Why the results will differ from the paper
Your numerical results in Tables 1 to 4 will be different from the paper's values.
This is expected and is proof that you ran the experiment independently.
The reasons include:
1. Only 25C data — no variation across extreme temperature conditions
2. Specific discharge currents only — a different operational subset
3. Smaller dataset relative to the full multi-temperature collection
4. Different optimizer (L-BFGS vs Adam) produces different convergence paths

The trend should match: neural networks outperform ensemble methods, and
wider/deeper networks achieve better SOC prediction accuracy.

---

## 5. Requirements

### MATLAB Toolboxes
| Toolbox                                 | Minimum Version | Functions Used                                  |
|-----------------------------------------|-----------------|-------------------------------------------------|
| Statistics and Machine Learning Toolbox | R2021b          | fitrnet, fitlm, fitrtree, fitrensemble, templateTree |

The Deep Learning Toolbox is NOT required.
fitrnet() from the Statistics and Machine Learning Toolbox handles all neural network
training. This was a deliberate design decision to keep the project accessible on
standard institutional MATLAB installations.

### MATLAB Version
- Minimum required: R2021b (fitrnet with Activations parameter introduced here)
- Recommended: R2022b or R2023a

### Verify your toolbox in MATLAB
```matlab
ver('stats')
% Should show: Statistics and Machine Learning Toolbox  Version 12.x or higher
```

---

## 6. Dataset Setup

### Download
LG 18650HG2 Li-ion Battery Data — publicly available on Kaggle:
https://www.kaggle.com/datasets/aditya9790/lg-18650hg2-liion-battery-data

Original dataset credit: Dr. Phillip Kollmeyer, McMaster University, Canada.
Full dataset citation: Kollmeyer, P. (2018). Mendeley Data. See Section 17.

### Required Folder Structure
After downloading and extracting from Kaggle your folder must look like this:

```
Dataset_Li-ion/
|-- 10degC/
|   |-- 571_Mixed1.csv
|   |-- 571_Mixed2.csv
|   +-- ...
|-- 25degC/          <-- ONLY THIS FOLDER IS READ BY THIS PROJECT
|   |-- 551_Charge1.csv
|   |-- 551_HWFET.csv
|   |-- 551_LA92.csv
|   |-- 556_Charge1.csv
|   |-- 556_Mixed1.csv
|   +-- ... (all other CSV files)
|-- 40degC/
|   +-- ...
|-- n10degC/
|   +-- ...
+-- n20degC/
    +-- ...
```

Only the 25degC subfolder is read. All other temperature folders are ignored.

### CSV File Format
Each CSV file has this internal structure:
- Several metadata and comment lines at the top   — automatically skipped
- A header row: Time Stamp, Voltage, Current, Temperature, Capacity, SOC, ...
- A units row immediately after the header        — automatically skipped
- Data rows from line 3 onwards

All known quirks are handled automatically by the parser:
- Null characters (char 0) stripped from all lines
- Ragged rows with inconsistent column counts padded or trimmed to match header
- Column names with spaces preserved using VariableNamingRule preserve
- Ambiguous datetime format pinned to MM/dd/uuuu hh:mm:ss aa
- Prog Time column type conflicts resolved by dropping before stacking
- Entirely NaN trailing columns dropped automatically

---

## 7. File Structure

```
your_project_folder/
|
|-- main_soc_analysis.m     <- ENTRY POINT: run this to execute everything
|
|-- s01_load_data.m         <- Stage 1: Load, filter, scale, split dataset
|-- s02_ensemble_models.m   <- Stage 2: Linear regression + ensemble trees
|-- s03_neural_networks.m   <- Stage 3: 15 NN variants (Tables 2 and 3)
|-- s04_wide_trilayer.m     <- Stage 4: Wide tri-layered FFNN (Table 4)
|-- s05_figures.m           <- Stage 5: All 13 figures including Fig 6 OCV
|-- s06_tables.m            <- Stage 6: All 5 tables printed and saved as CSV
|
|-- soc_estimation.m        <- Standalone single-file version
|
|-- Dataset_Li-ion/         <- You provide this from Kaggle
|   +-- 25degC/
|       +-- *.csv
|
+-- README.md               <- This file
```

### Detailed Role of Each File

#### main_soc_analysis.m
Master runner. Calls all 6 stage scripts in order. Variables are shared across
all stages through the MATLAB workspace. Do not run clear between stages or
variables from earlier stages will be lost and subsequent stages will error.

#### s01_load_data.m
- Reads all CSV files from Dataset_Li-ion/25degC/
- Immediately drops all columns except Voltage, Current, Temperature, SOC,
  and Capacity per file before stacking (prevents the Prog Time vertcat crash)
- Filters rows to current near 0.10A, 0.75A, 1.5A, 2.0A, or 3.0A (+/-0.20A)
- Applies MinMax scaling to [0, 1] for all features and target column
- Performs 70/30 train/test split with rng(42) for full reproducibility
- Caps total rows at 500,000 (actual count depends on 25C folder content)
- Key outputs to workspace:
    X_train, X_test          scaled feature matrices
    y_train, y_test          scaled target vectors
    X_train_raw, X_test_raw  unscaled features for plotting
    y_train_raw, y_test_raw  unscaled targets for plotting
    yMin, yMax               scaling bounds for inverse transform
    targetCol                name of target column used (SOC or Capacity)

#### s02_ensemble_models.m
- Trains: Linear Regression, Decision Tree, Bagged Trees (30 learners),
  and Boosted Trees (30 learners at learning rate 0.01)
- Computes MAE, MSE, RMSE, R-squared on both train and test sets
- Local function computeMetrics placed at the bottom of the file
  (MATLAB rule: all local functions must appear after all executable code)
- Key outputs: results_train, results_test, ensemble_preds_test

#### s03_neural_networks.m
- Trains 15 neural networks: 5 architectures x 3 activation functions
- Architectures: Narrow[10], Medium[25], Wide[100], Bi-layer[10,10],
  Tri-layer[10,10,10]
- Activations tested per architecture: ReLU, Tanh, Sigmoid
- Key outputs: nn_results_train, nn_results_test, nn_labels

#### s04_wide_trilayer.m
- Trains 3 configurations of the wide tri-layered FFNN
    Config a: [100, 100, 100] at 1500 iterations
    Config b: [100, 200, 100] at 2000 iterations  (best model)
    Config c: [100, 100, 100] at 1000 iterations
- Best model stored separately for all downstream plotting
- Key outputs: wide_results_train, wide_results_test, wide_labels,
  best_net, best_pred_test, best_pred_train, best_pred_real, best_true_real

#### s05_figures.m
- Generates 13 figures: 12 matching the paper plus Fig 6 OCV reference curve
- Saves each figure as a PNG in the MATLAB working directory
- Fig 6 uses a 6th-order polynomial OCV-SOC model for LiNMC chemistry —
  it is a theoretical reference figure, not derived from the loaded dataset
- Does not modify any variables set by earlier stages

#### s06_tables.m
- Prints all 5 tables to the MATLAB Command Window with aligned columns
- Saves all 5 as CSV files in the MATLAB working directory
- Tables 1 to 4 use your independently computed results from the run
- Table 5 values are fixed from the paper and clearly labelled as reference

#### soc_estimation.m
Standalone single-file version of the complete pipeline. Trains and evaluates
the wide tri-layered FFNN only with no comparison models. Loads all temperature
folders and all currents without filtering. Does not depend on any stage files.
Useful for quick demonstration or testing outside the main pipeline.

---

## 8. How to Run

### Step 1 — Open MATLAB and set working directory
```matlab
cd 'C:\Users\YourName\Documents\SOC_Project'
```

### Step 2 — Confirm your dataset path
Open s01_load_data.m and check or update line 13:
```matlab
DATASET_ROOT = './Dataset_Li-ion';
% Or use a full absolute path:
DATASET_ROOT = 'C:\Users\YourName\Documents\Dataset_Li-ion';
```

### Step 3 — Run the master script
```matlab
run('main_soc_analysis.m')
```
Or open main_soc_analysis.m in the MATLAB Editor and press F5.

### Step 4 — Monitor progress in the Command Window
You will see each stage announce itself and print progress:

    --- Stage 1: Loading data ---
    Found 25 CSV files in '25degC'
      Loaded: 551_Charge1.csv    146 rows (running total: 146)
      ...
    Final dataset: XXXXX rows  |  Target column: 'SOC'
    Stage 1 complete.

    --- Stage 2: Ensemble models ---
      Training Linear Regression...
      Training Decision Tree...
      ...
    Stage 2 complete.

    --- Stage 3: Neural Network variants ---
      Training Narrow (ReLU)...
      ...

### Step 5 — Check outputs when finished
Your project folder will contain 13 PNG figure files and 5 CSV table files.
All metrics are also printed in full to the Command Window.

### Expected Runtime
| Stage | Description                             | Approximate Time   |
|-------|-----------------------------------------|--------------------|
| s01   | Load and filter up to 500,000 rows      | 2 to 5 minutes     |
| s02   | 4 ensemble models                       | 2 to 5 minutes     |
| s03   | 15 NN variants at 1000 iterations each  | 15 to 30 minutes   |
| s04   | 3 wide tri-layer configurations         | 10 to 20 minutes   |
| s05   | 13 figures including convergence curves | 5 to 15 minutes    |
| s06   | 5 tables                                | Under 1 second     |
| Total |                                         | 35 to 75 minutes   |

For a quick test run reduce MAX_ROWS to 20000 in s01_load_data.m.
This brings total runtime down to approximately 10 to 15 minutes.

---

## 9. Configuration and Tuning

All key settings are at the top of s01_load_data.m:

```matlab
DATASET_ROOT     = './Dataset_Li-ion';             % path to dataset root
TARGET_SUBDIR    = '25degC';                       % temperature subfolder
MAX_ROWS         = 500000;                         % row cap
CURRENT_TARGETS  = [0.75, 0.10, 1.5, 2.0, 3.0];  % discharge currents in Amps
CURRENT_BAND     = 0.20;                           % +/- tolerance in Amps
```

### Understanding the Current Filter
Each value in CURRENT_TARGETS defines the centre of an acceptance band.
A row is kept if: abs(Current) is within CURRENT_TARGETS(i) +/- CURRENT_BAND

With the current settings the filter accepts rows where |Current| falls in:
- 0.55A to 0.95A   (centred on 0.75A)
- 0.00A to 0.30A   (centred on 0.10A)
- 1.30A to 1.70A   (centred on 1.50A)
- 1.80A to 2.20A   (centred on 2.00A)
- 2.80A to 3.20A   (centred on 3.00A)
All other current values are discarded.

### Speed vs Accuracy
| MAX_ROWS  | Approx. Total Runtime | Result Quality          |
|-----------|-----------------------|-------------------------|
| 10,000    | 5 to 10 minutes       | Lower — quick testing   |
| 20,000    | 10 to 15 minutes      | Reasonable              |
| 50,000    | 15 to 25 minutes      | Good                    |
| 200,000   | 25 to 45 minutes      | High                    |
| 500,000   | 35 to 75 minutes      | Best achievable         |

### If actual row count is lower than 500,000
The 25C folder has a finite amount of data matching these currents. If Stage 1
prints a lower total that is completely normal — the cap was simply never reached.

To get more rows if your supervisor requires it:

Option A — Widen the current filter to accept all discharge conditions:
```matlab
CURRENT_TARGETS = [1, 2, 3, 4, 5];
CURRENT_BAND    = 0.50;
```

Option B — Use a different temperature subfolder:
```matlab
TARGET_SUBDIR = '10degC';   % options: 10degC, 40degC, n10degC, n20degC
```

Option C — Remove row cap entirely:
```matlab
MAX_ROWS = Inf;
```

### If no rows load at all (filter too restrictive)
Widen CURRENT_BAND:
```matlab
CURRENT_BAND = 0.40;
```
Or check what currents actually exist in your files:
```matlab
csvFiles = dir('Dataset_Li-ion/25degC/*.csv');
f = fullfile('Dataset_Li-ion/25degC', csvFiles(1).name);
t = readtable(f, 'NumHeaderLines', 3);
disp(unique(round(abs(t{:,4}), 2)))
```

---

## 10. Output Files

### PNG Figures (13 files)
| Filename                          | Description                                        |
|-----------------------------------|----------------------------------------------------|
| Fig2_training_data_profile.png    | 6-panel feature profile of training data           |
| Fig4_ensemble_training.png        | MAE and RMSE — ensemble models on training set     |
| Fig5_ensemble_testing.png         | MAE and RMSE — ensemble models on test set         |
| Fig6_OCV_vs_SOC.png               | OCV vs SOC theoretical reference curve             |
| Fig10_all_nn_rmse.png             | RMSE bar chart across all 15 NN model variants     |
| Fig11_residual_wide_nn.png        | Residual plot — wide single-layer NN               |
| Fig12_residual_trilayer.png       | Residual plot — wide tri-layered FFNN best model   |
| Fig13_singlelayer_convergence.png | MSE convergence curve — single-layer FFNN          |
| Fig14_singlelayer_regression.png  | Predicted vs actual — single-layer FFNN            |
| Fig15_trilayer_convergence.png    | MSE convergence curve — tri-layered FFNN           |
| Fig16_trilayer_regression.png     | Predicted vs actual — tri-layered FFNN best model  |
| Fig17_error_hist_singlelayer.png  | Error histogram 20 bins — single-layer FFNN        |
| Fig18_error_hist_trilayer.png     | Error histogram 20 bins — tri-layered FFNN         |

### CSV Tables (5 files)
| Filename                       | Values                              |
|--------------------------------|-------------------------------------|
| Table1_ensemble_results.csv    | YOUR independently computed results |
| Table2_nn_training.csv         | YOUR independently computed results |
| Table3_nn_testing.csv          | YOUR independently computed results |
| Table4_wide_trilayer.csv       | YOUR independently computed results |
| Table5_method_comparison.csv   | Fixed from paper — cite accordingly |

---

## 11. Figures Generated

| Figure                   | Paper Fig | Description                                     | Data Source              |
|--------------------------|-----------|-------------------------------------------------|--------------------------|
| Fig 2 training profile   | Fig. 2    | 6-panel: Voltage, Current, Temp, AvgV, AvgI, SOC | Your dataset             |
| Fig 4 ensemble training  | Fig. 4    | MAE and RMSE scatter — ensemble, training set   | Your results             |
| Fig 5 ensemble testing   | Fig. 5    | MAE and RMSE scatter — ensemble, test set       | Your results             |
| Fig 6 OCV vs SOC         | Fig. 6    | Theoretical OCV vs SOC non-linear S-curve       | LiNMC polynomial model   |
| Fig 10 NN RMSE bar       | Fig. 10   | RMSE bar chart across all 15 NN variants        | Your results             |
| Fig 11 residual wide NN  | Fig. 11   | Residual plot — wide single-layer neural network| Your results             |
| Fig 12 residual trilayer | Fig. 12   | Residual plot — wide tri-layered FFNN           | Your results             |
| Fig 13 convergence s-l   | Fig. 13   | MSE vs iterations — single-layer FFNN           | Your results             |
| Fig 14 regression s-l    | Fig. 14   | Predicted vs actual — single-layer FFNN         | Your results             |
| Fig 15 convergence tri   | Fig. 15   | MSE vs iterations — tri-layered FFNN            | Your results             |
| Fig 16 regression tri    | Fig. 16   | Predicted vs actual — tri-layered FFNN          | Your results             |
| Fig 17 histogram s-l     | Fig. 17   | Error histogram 20 bins — single-layer FFNN     | Your results             |
| Fig 18 histogram tri     | Fig. 18   | Error histogram 20 bins — tri-layered FFNN      | Your results             |

Note on Fig 6:
This is a theoretical reference illustration reproduced using a standard 6th-order
polynomial OCV-SOC model for LiNMC battery chemistry. It is NOT generated from
your loaded dataset. It corresponds to Fig 6 in the paper which was originally
sourced from Berglund et al. (2019). When using this figure in your report,
label it as a reference illustration and cite Berglund et al. (2019) alongside
Ofoegbu (2025).

---

## 12. Tables Generated

| Table   | Paper Table | Contents                                          | Values                          |
|---------|-------------|---------------------------------------------------|---------------------------------|
| Table 1 | Table 1     | Linear Reg, Tree, Bagged, Boosted — MAE MSE RMSE R2 | Your independently computed   |
| Table 2 | Table 2     | All 15 NN variants — training metrics             | Your independently computed     |
| Table 3 | Table 3     | All 15 NN variants — testing metrics              | Your independently computed     |
| Table 4 | Table 4     | Wide tri-layer 3 configs — train and test metrics | Your independently computed     |
| Table 5 | Table 5     | Comparison with 12 related methods from literature| Fixed from paper — cite this    |

Tables 1 to 4 contain your independently computed values from your specific data
subset. They will naturally differ from the paper's values, which is expected and
demonstrates you ran the experiment yourself.

Table 5 values are taken directly from the paper. Present them in your report as
a literature comparison table with a full citation to Ofoegbu (2025). Never
present Table 5 values as your own experimental results.

---

## 13. Models Implemented

### Ensemble Models (s02_ensemble_models.m)
| Model             | MATLAB Function | Key Parameters                                 |
|-------------------|-----------------|------------------------------------------------|
| Linear Regression | fitlm           | Default settings                               |
| Decision Tree     | fitrtree        | MinLeafSize = 8                                |
| Bagged Trees      | fitrensemble    | Method=Bag, 30 cycles, MinLeafSize=8           |
| Boosted Trees     | fitrensemble    | Method=LSBoost, 30 cycles, LearnRate=0.01      |

### Neural Networks (s03_neural_networks.m and s04_wide_trilayer.m)
All networks use fitrnet() — Statistics and Machine Learning Toolbox only.
No Deep Learning Toolbox required.

| Model                        | Architecture    | Activations Tested   | Iterations |
|------------------------------|-----------------|----------------------|------------|
| Narrow FFNN                  | [10]            | ReLU, Tanh, Sigmoid  | 1000       |
| Medium FFNN                  | [25]            | ReLU, Tanh, Sigmoid  | 1000       |
| Wide FFNN                    | [100]           | ReLU, Tanh, Sigmoid  | 1000       |
| Bi-layered FFNN              | [10, 10]        | ReLU, Tanh, Sigmoid  | 1000       |
| Tri-layered FFNN             | [10, 10, 10]    | ReLU, Tanh, Sigmoid  | 1000       |
| Wide Tri-layer config a      | [100, 100, 100] | ReLU                 | 1500       |
| Wide Tri-layer config b BEST | [100, 200, 100] | ReLU                 | 2000       |
| Wide Tri-layer config c      | [100, 100, 100] | ReLU                 | 1000       |

Config b [100, 200, 100] at 2000 iterations is the paper's recommended best model.
This is the architecture to highlight in your report as the primary result.

### Features and Target Variable
| Role              | CSV Column  | Description                           |
|-------------------|-------------|---------------------------------------|
| Feature           | Voltage     | Cell terminal voltage in Volts        |
| Feature           | Current     | Charge or discharge current in Amps   |
| Feature           | Temperature | Cell surface temperature in Celsius   |
| Target (primary)  | SOC         | State of Charge — used if available   |
| Target (fallback) | Capacity    | Used only if SOC column not found     |

### Data Preprocessing Pipeline
1. Drop all non-essential columns per file before stacking (prevents type mismatches)
2. Filter rows to accepted current bands
3. Remove rows with any NaN in Voltage, Current, Temperature, or target column
4. Apply MinMax scaling to [0, 1]: scaled = (x - min) / (max - min)
5. Shuffle with rng(42) then split 70% training / 30% testing
6. Store yMin and yMax for inverse-transform when plotting real-unit results

---

## 14. Known Issues and Fixes Applied

All issues below were encountered during development and are already fixed.
No action is required — everything is handled automatically in the code.

| Error Message                                          | Root Cause                                              | Fix Applied                                                                   |
|--------------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Error concatenating Prog Time using VERTCAT            | Prog Time stored as different types across CSV files    | All non-essential columns dropped per file before vertcat is called           |
| Arrays have incompatible sizes                         | Ragged CSV rows with inconsistent column counts         | Parser pads or trims every row to match the header column count exactly       |
| Column headers modified to valid MATLAB identifiers    | MATLAB auto-renames columns containing spaces           | VariableNamingRule set to preserve in all detectImportOptions calls           |
| DATETIME matched both MM/dd/uuuu and dd/MM/uuuu        | MATLAB cannot resolve ambiguous date format             | setvaropts pins Time Stamp to MM/dd/uuuu hh:mm:ss aa explicitly              |
| Function definitions must appear at end of file        | computeMetrics function placed before executable code   | Function moved to bottom of s02_ensemble_models.m after all executable code  |
| All table variables must have the same number of rows  | wide_labels had 3 entries but data had 6 rows           | wide_labels doubled using [wide_labels; wide_labels] before table() call     |

---

## 15. Troubleshooting

### Dataset folder not found
```matlab
pwd                                    % check current working directory
isfolder('./Dataset_Li-ion')           % should return 1
isfolder('./Dataset_Li-ion/25degC')    % should return 1
dir('./Dataset_Li-ion/25degC')         % should list CSV files
```
If any return 0 update DATASET_ROOT in s01_load_data.m to the full absolute path.

### Stage 1 reports zero rows loaded
The current filter found no matching rows in the CSV files.
```matlab
% In s01_load_data.m try widening the tolerance:
CURRENT_BAND = 0.40;
```
Or diagnose what currents exist in your data:
```matlab
csvFiles = dir('Dataset_Li-ion/25degC/*.csv');
f = fullfile('Dataset_Li-ion/25degC', csvFiles(1).name);
t = readtable(f, 'NumHeaderLines', 3);
disp(unique(round(abs(t{:,4}), 2)))
```

### fitrnet is not recognised
Your MATLAB version is below R2021b or the toolbox is not installed:
```matlab
ver('stats')    % needs Version 12.0 (R2021b) or later
```
Contact your institution IT team if the toolbox is missing.

### Re-run a single stage without restarting everything
All workspace variables persist between stages:
```matlab
run('s04_wide_trilayer.m')   % re-run best model only
run('s05_figures.m')         % re-generate all figures
run('s06_tables.m')          % re-print and re-save all tables
```

### Start completely fresh
```matlab
clc; clear; close all;
run('main_soc_analysis.m');
```

### Training is too slow
Reduce MAX_ROWS in s01_load_data.m:
```matlab
MAX_ROWS = 20000;   % approximately 10 to 15 minutes total
```

---

## 16. Notes for Report Writing

### Describing your dataset in your report
    "Experiments were conducted using the publicly available LG 18650HG2 Li-ion
    battery dataset (Kollmeyer, 2018). A subset recorded at 25 degrees C was used,
    filtered to discharge currents of approximately 0.10A, 0.75A, 1.5A, 2.0A, and
    3.0A (tolerance +/-0.20A), with a maximum of 500,000 rows. Data was split 70/30
    into training and test sets with a fixed random seed of 42 for reproducibility.
    All input features (Voltage, Current, Temperature) and the SOC target were
    normalised to [0, 1] using MinMax scaling prior to model training."

### Describing your neural network model
    "All neural network models were implemented using the fitrnet function from
    MATLAB's Statistics and Machine Learning Toolbox (R2022b), which uses L-BFGS
    optimisation internally. The best performing configuration was a wide tri-layered
    FFNN with layer sizes [100, 200, 100] and ReLU activation, trained for 2000
    iterations with no regularisation (lambda = 0)."

### Explaining why your results differ from the paper
    "Results differ from Ofoegbu (2025) as expected, due to the use of a
    single-temperature, filtered-current subset and a different optimisation
    algorithm (L-BFGS vs Adam). The trend is consistent with the paper: the
    wide tri-layered FFNN outperforms all ensemble and shallower NN baselines."

### Key figures to prioritise in your report
| Priority | Figure   | Why It Matters                                              |
|----------|----------|-------------------------------------------------------------|
| 1        | Fig 6    | Motivates why a model is needed (flat OCV region)           |
| 2        | Fig 2    | Shows the characteristics of your training dataset         |
| 3        | Fig 16   | Demonstrates your best model performance visually           |
| 4        | Fig 18   | Shows error distribution of your best model                 |
| 5        | Fig 10   | Compares all NN variants and justifies architecture choice  |

### Key tables to prioritise in your report
| Priority | Table   | Why It Matters                                              |
|----------|---------|-------------------------------------------------------------|
| 1        | Table 4 | Justifies the wide tri-layer architecture as the best model |
| 2        | Table 1 | Justifies moving beyond simple regression to neural nets    |
| 3        | Table 5 | Positions your work in context of existing literature       |

---

## 17. References

1. Ofoegbu, E.O. (2025).
   State of charge (SOC) estimation in electric vehicle (EV) battery management
   systems using ensemble methods and neural networks.
   Journal of Energy Storage, 114, 115833.
   https://doi.org/10.1016/j.est.2025.115833

2. Kollmeyer, P. (2018).
   LG 18650HG2 Li-ion Battery Data.
   Mendeley Data, V3.
   https://data.mendeley.com/datasets/cp3473x7xv/3

3. Berglund, F., Bosstrom, C., and Soussou, R. (2019).
   Modelling of lithium-ion battery for simulation of hybrid electric vehicle.
   Cited in Ofoegbu (2025) as reference [32] — original source of the
   OCV vs SOC relationship illustrated in Fig 6 of this project.

4. MathWorks (2021).
   fitrnet — Train neural network for regression.
   MATLAB Statistics and Machine Learning Toolbox, R2021b.
   https://www.mathworks.com/help/stats/fitrnet.html

---

All code and results in this project are independently produced.
The reference paper methodology was used as a framework only.
All numerical results in Tables 1 to 4 and Figures 2 to 18 are original outputs
generated by running the MATLAB code on the specified data subset.