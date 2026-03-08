%% =============================================================================
%% main_soc_analysis.m  — Master runner
%%
%% Comparison of Ensemble Methods and Feedforward Neural Networks
%% for State of Charge Estimation in Li-ion Electric Vehicle Batteries
%%
%% Based on: Ofoegbu, Journal of Energy Storage, 2025
%%
%% Execution order:
%%   1. s01_load_data.m      — Load and preprocess the dataset
%%   2. s02_ensemble_models.m — Linear regression + Decision Tree + Ensemble
%%   3. s03_neural_networks.m — 15 FFNN configurations (5 archs x 3 activations)
%%   4. s04_wide_trilayer.m  — Wide Tri-Layered FFNN (3 configurations)
%%   5. s05_figures.m        — All individual model figures
%%   6. s06_tables.m         — All individual model results tables
%%   7. s07_comparison.m     — Head-to-head comparison: Ensemble vs FFNN
%%
%% Requirements: Statistics and Machine Learning Toolbox (R2021b+)
%% =============================================================================
clc; clear; close all;

fprintf('============================================================\n');
fprintf('  SOC Estimation — LG HG2 Battery\n');
fprintf('  Comparison: Ensemble Methods vs Feedforward Neural Networks\n');
fprintf('  Based on: Ofoegbu, Journal of Energy Storage, 2025\n');
fprintf('============================================================\n\n');

%% ---- Create output subdirectories ----
if ~exist('figures', 'dir'), mkdir('figures'); end
if ~exist('tables',  'dir'), mkdir('tables');  end
fprintf('Output folders ready: figures/  tables/\n\n');

%% ---- Run each stage in order ----
run('s01_load_data.m');
run('s02_ensemble_models.m');
run('s03_neural_networks.m');
run('s04_wide_trilayer.m');
run('s05_figures.m');
run('s06_tables.m');
run('s07_comparison.m');     %% <-- comparison summary (new)

fprintf('\n============================================================\n');
fprintf('  All done!\n');
fprintf('  Tables:  tables/ folder — Table1–Table7 CSV files.\n');
fprintf('  Figures: figures/ folder — Figure1–Figure8 PNG files.\n');
fprintf('  Key output: tables/Table6_full_comparison_ranking.csv\n');
fprintf('              tables/Table7_head_to_head.csv\n');
fprintf('              figures/Figure7_ensemble_vs_ffnn_comparison.png\n');
fprintf('============================================================\n');