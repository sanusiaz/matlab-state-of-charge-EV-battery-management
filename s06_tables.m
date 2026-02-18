%% =============================================================================
%% s06_tables.m  — Print all tables from the paper to the Command Window
%%                 and save as CSV files for easy reference
%%
%% Tables reproduced:
%%   Table 1 — Ensemble model results (training + testing)
%%   Table 2 — NN training results (all variants x activations)
%%   Table 3 — NN testing results  (all variants x activations)
%%   Table 4 — Wide Tri-layered FFNN detailed results
%%   Table 5 — Comparison with related SOC methods (from paper, fixed values)
%% =============================================================================

fprintf('--- Stage 6: Generating all tables ---\n\n');

%% =====================================================================
%% TABLE 1 — Ensemble & Linear Regression Results
%% =====================================================================
fprintf('========================================================\n');
fprintf('TABLE 1 — Model Performance (Training & Testing)\n');
fprintf('========================================================\n');
fprintf('%-22s %-10s %-12s %-10s %-10s\n','Model','MAE','MSE','RMSE','R-sq');
fprintf('%s\n', repmat('-',1,68));
fprintf('TRAINING\n');
t1_train = [
    results_train.lr_mae,    results_train.lr_mse,    results_train.lr_rmse,    results_train.lr_r2;
    results_train.tree_mae,  results_train.tree_mse,  results_train.tree_rmse,  results_train.tree_r2;
    results_train.bag_mae,   results_train.bag_mse,   results_train.bag_rmse,   results_train.bag_r2;
    results_train.boost_mae, results_train.boost_mse, results_train.boost_rmse, results_train.boost_r2
];
t1_names = {'Linear Regression','Decision Tree','Ensemble (Bagged)','Ensemble (Boosted)'};
for i = 1:4
    fprintf('%-22s %-10.7f %-12.7f %-10.7f %-10.4f\n', ...
            t1_names{i}, t1_train(i,1), t1_train(i,2), t1_train(i,3), t1_train(i,4));
end
fprintf('TESTING\n');
t1_test = [
    results_test.lr_mae,    results_test.lr_mse,    results_test.lr_rmse,    results_test.lr_r2;
    results_test.tree_mae,  results_test.tree_mse,  results_test.tree_rmse,  results_test.tree_r2;
    results_test.bag_mae,   results_test.bag_mse,   results_test.bag_rmse,   results_test.bag_r2;
    results_test.boost_mae, results_test.boost_mse, results_test.boost_rmse, results_test.boost_r2
];
for i = 1:4
    fprintf('%-22s %-10.7f %-12.7f %-10.7f %-10.4f\n', ...
            t1_names{i}, t1_test(i,1), t1_test(i,2), t1_test(i,3), t1_test(i,4));
end

% Save Table 1 as CSV
T1 = table([t1_names';t1_names'], ...
    [t1_train(:,1);t1_test(:,1)], ...
    [t1_train(:,2);t1_test(:,2)], ...
    [t1_train(:,3);t1_test(:,3)], ...
    [t1_train(:,4);t1_test(:,4)], ...
    'VariableNames', {'Model','MAE','MSE','RMSE','R_squared'});
T1.Phase = [repmat({'Training'},4,1); repmat({'Testing'},4,1)];
writetable(T1, 'Table1_ensemble_results.csv');
fprintf('\n  Table 1 saved: Table1_ensemble_results.csv\n\n');


%% =====================================================================
%% TABLE 2 — Neural Network TRAINING results
%% =====================================================================
fprintf('========================================================\n');
fprintf('TABLE 2 — Neural Network Training Results\n');
fprintf('========================================================\n');
fprintf('%-16s %-10s %-10s %-12s %-10s %-10s\n', ...
        'Model','Activation','MAE','MSE','RMSE','R-sq');
fprintf('%s\n', repmat('-',1,72));
for r = 1:15
    fprintf('%-16s %-10s %-10.9f %-12.9f %-10.9f %-10.6f\n', ...
            nn_labels{r,1}, nn_labels{r,2}, ...
            nn_results_train(r,1), nn_results_train(r,2), ...
            nn_results_train(r,3), nn_results_train(r,4));
end

T2 = array2table(nn_results_train, 'VariableNames', {'MAE','MSE','RMSE','R_squared'});
T2.Model      = nn_labels(:,1);
T2.Activation = nn_labels(:,2);
T2 = T2(:, [5 6 1 2 3 4]);
writetable(T2, 'Table2_nn_training.csv');
fprintf('\n  Table 2 saved: Table2_nn_training.csv\n\n');


%% =====================================================================
%% TABLE 3 — Neural Network TESTING results
%% =====================================================================
fprintf('========================================================\n');
fprintf('TABLE 3 — Neural Network Testing Results\n');
fprintf('========================================================\n');
fprintf('%-16s %-10s %-10s %-12s %-10s %-10s\n', ...
        'Model','Activation','MAE','MSE','RMSE','R-sq');
fprintf('%s\n', repmat('-',1,72));
for r = 1:15
    fprintf('%-16s %-10s %-10.9f %-12.9f %-10.9f %-10.6f\n', ...
            nn_labels{r,1}, nn_labels{r,2}, ...
            nn_results_test(r,1), nn_results_test(r,2), ...
            nn_results_test(r,3), nn_results_test(r,4));
end

T3 = array2table(nn_results_test, 'VariableNames', {'MAE','MSE','RMSE','R_squared'});
T3.Model      = nn_labels(:,1);
T3.Activation = nn_labels(:,2);
T3 = T3(:, [5 6 1 2 3 4]);
writetable(T3, 'Table3_nn_testing.csv');
fprintf('\n  Table 3 saved: Table3_nn_testing.csv\n\n');


%% =====================================================================
%% TABLE 4 — Wide Tri-layered FFNN detailed results
%% =====================================================================
fprintf('========================================================\n');
fprintf('TABLE 4 — Wide Tri-layered FFNN Results\n');
fprintf('========================================================\n');
fprintf('%-35s %-10s %-12s %-10s %-10s\n','Parameters','MAE','MSE','RMSE','R-sq');
fprintf('%s\n', repmat('-',1,80));
fprintf('TRAINING\n');
for c = 1:3
    fprintf('%-35s %-10.9f %-12.9f %-10.9f %-10.6f\n', ...
            wide_labels{c}, wide_results_train(c,1), wide_results_train(c,2), ...
            wide_results_train(c,3), wide_results_train(c,4));
end
fprintf('TESTING\n');
for c = 1:3
    fprintf('%-35s %-10.9f %-12.9f %-10.9f %-10.6f\n', ...
            wide_labels{c}, wide_results_test(c,1), wide_results_test(c,2), ...
            wide_results_test(c,3), wide_results_test(c,4));
end

% wide_labels has 3 entries; stack it twice to match 6 rows (3 train + 3 test)
wide_labels_doubled = [wide_labels; wide_labels];
T4 = table(wide_labels_doubled, ...
    [wide_results_train(:,1);wide_results_test(:,1)], ...
    [wide_results_train(:,2);wide_results_test(:,2)], ...
    [wide_results_train(:,3);wide_results_test(:,3)], ...
    [wide_results_train(:,4);wide_results_test(:,4)], ...
    'VariableNames', {'Parameters','MAE','MSE','RMSE','R_squared'});
T4.Phase = [repmat({'Training'},3,1); repmat({'Testing'},3,1)];
writetable(T4, 'Table4_wide_trilayer.csv');
fprintf('\n  Table 4 saved: Table4_wide_trilayer.csv\n\n');


%% =====================================================================
%% TABLE 5 — Comparison with related SOC methods (paper values, fixed)
%% =====================================================================
fprintf('========================================================\n');
fprintf('TABLE 5 — Performance Comparison with Related Methods\n');
fprintf('========================================================\n');

t5_methods = {
    'Kalman filter (UKF)';
    'Kalman filter (CDKF)';
    'Particle filtering (PF)';
    'Kalman filter (AEKF)';
    'Adaptive particle filtering (APF)';
    'Proposed FFNN (wide)  <-- THIS STUDY';
    'Deep neural network (DNN)';
    'CNN (U-NET)';
    'LSTM';
    'LSTM-AHIF';
    'Recurrent neural network (RNN)';
    'GRU-RNN';
    'DAE-GRU'
};
t5_errors = [1.05; 0.96; 0.20; 7.26; 4.62; 0.88; 1.10; 1.50; 2.36; 1.18; 2.50; 2.53; 1.59];
t5_method_col = {'MAE %';'MAE %';'MAE %';'MAE %';'MAE %';'MAE %';'MAE %'; ...
                 'MAE %';'MAE %';'MAE %';'MAE %';'MAE %';'MAE %'};

fprintf('%-42s %-10s %-10s\n','Method','Error (%)', 'Metric');
fprintf('%s\n', repmat('-',1,65));
for i = 1:numel(t5_methods)
    if i == 6
        fprintf('>>> %-39s %-10.2f %-10s <<<\n', t5_methods{i}, t5_errors(i), t5_method_col{i});
    else
        fprintf('    %-39s %-10.2f %-10s\n', t5_methods{i}, t5_errors(i), t5_method_col{i});
    end
end

T5 = table(t5_methods, t5_errors, t5_method_col, ...
    'VariableNames', {'Method','Error_Percent','Error_Metric'});
writetable(T5, 'Table5_method_comparison.csv');
fprintf('\n  Table 5 saved: Table5_method_comparison.csv\n\n');

fprintf('Stage 6 complete — all tables printed and saved.\n');