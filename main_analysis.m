%% =============================================================================
%% main_soc_analysis.m  — Master runner
%%
%% Runs the full SOC estimation pipeline and generates ALL tables and figures
%% from the paper (Ofoegbu, Journal of Energy Storage, 2025).
%%
%% Execution order:
%%   1. s01_load_data.m          — Load a small, fast subset of the dataset
%%   2. s02_ensemble_models.m    — Linear regression + Decision tree + Ensemble
%%   3. s03_neural_networks.m    — Narrow / Medium / Wide / Bi / Tri-layer FFNNs
%%   4. s04_wide_trilayer.m      — Detailed wide tri-layered FFNN (paper's best model)
%%   5. s05_figures.m            — All figures from the paper
%%   6. s06_tables.m             — All formatted tables from the paper
%%
%% Requirements: Statistics and Machine Learning Toolbox (R2021b+)
%% =============================================================================
clc; clear; close all;

fprintf('============================================================\n');
fprintf('  SOC Estimation — LG HG2 Battery (School Project)\n');
fprintf('  Based on: Ofoegbu, Journal of Energy Storage, 2025\n');
fprintf('============================================================\n\n');

%% ---- Run each stage in order ----
run('s01_load_data.m');
run('s02_ensemble_models.m');
run('s03_neural_networks.m');
run('s04_wide_trilayer.m');
run('s05_figures.m');
run('s06_tables.m');

fprintf('\n============================================================\n');
fprintf('  All done! Check this folder for saved PNG and table files.\n');
fprintf('============================================================\n');