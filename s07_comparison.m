%% =============================================================================
%% s07_comparison.m  — HEAD-TO-HEAD COMPARISON: Ensemble vs FFNN
%%
%% This script is the central output of the comparison study.
%% It pulls together results from all previous stages and answers
%% the core research question:
%%
%%   "Which family of models performs better for SOC estimation —
%%    ensemble methods or feedforward neural networks?"
%%
%% It does three things:
%%   1. Identifies the best model from each family (test MAE)
%%   2. Prints a structured head-to-head comparison table
%%   3. Generates Figure 7 — the comparison bar chart (MAE + R²)
%%
%% Requires: results from s02, s03, s04, s05 to already be in workspace
%% =============================================================================

fprintf('--- Stage 7: Head-to-Head Comparison (Ensemble vs FFNN) ---\n\n');

%% =====================================================================
%% 1. COLLECT BEST RESULT FROM EACH FAMILY
%% =====================================================================

%% --- Ensemble family: pick lowest test MAE among the 4 models ---
ensemble_names = {'Linear Regression', 'Decision Tree', ...
                  'Bagged Trees', 'Boosted Trees'};
ensemble_test_mae  = [results_test.lr_mae,   results_test.tree_mae, ...
                      results_test.bag_mae,  results_test.boost_mae];
ensemble_test_mse  = [results_test.lr_mse,   results_test.tree_mse, ...
                      results_test.bag_mse,  results_test.boost_mse];
ensemble_test_rmse = [results_test.lr_rmse,  results_test.tree_rmse, ...
                      results_test.bag_rmse, results_test.boost_rmse];
ensemble_test_r2   = [results_test.lr_r2,    results_test.tree_r2, ...
                      results_test.bag_r2,   results_test.boost_r2];

[best_ens_mae, best_ens_idx] = min(ensemble_test_mae);
best_ens_name  = ensemble_names{best_ens_idx};
best_ens_mse   = ensemble_test_mse(best_ens_idx);
best_ens_rmse  = ensemble_test_rmse(best_ens_idx);
best_ens_r2    = ensemble_test_r2(best_ens_idx);

%% --- Standard FFNN family: pick lowest test MAE among 15 configs ---
[best_ffnn_mae, best_ffnn_idx] = min(nn_results_test(:,1));
best_ffnn_name  = sprintf('%s (%s)', nn_labels{best_ffnn_idx,1}, ...
                                     nn_labels{best_ffnn_idx,2});
best_ffnn_mse   = nn_results_test(best_ffnn_idx, 2);
best_ffnn_rmse  = nn_results_test(best_ffnn_idx, 3);
best_ffnn_r2    = nn_results_test(best_ffnn_idx, 4);

%% --- Wide Tri-Layer FFNN family: pick lowest test MAE among 3 configs ---
[best_wide_mae, best_wide_idx] = min(wide_results_test(:,1));
best_wide_name  = wide_labels{best_wide_idx};
best_wide_mse   = wide_results_test(best_wide_idx, 2);
best_wide_rmse  = wide_results_test(best_wide_idx, 3);
best_wide_r2    = wide_results_test(best_wide_idx, 4);

%% --- Overall best FFNN: compare standard vs wide ---
if best_ffnn_mae <= best_wide_mae
    overall_ffnn_name = best_ffnn_name;
    overall_ffnn_mae  = best_ffnn_mae;
    overall_ffnn_mse  = best_ffnn_mse;
    overall_ffnn_rmse = best_ffnn_rmse;
    overall_ffnn_r2   = best_ffnn_r2;
else
    overall_ffnn_name = best_wide_name;
    overall_ffnn_mae  = best_wide_mae;
    overall_ffnn_mse  = best_wide_mse;
    overall_ffnn_rmse = best_wide_rmse;
    overall_ffnn_r2   = best_wide_r2;
end

%% =====================================================================
%% 2. PRINT HEAD-TO-HEAD COMPARISON TABLE
%% =====================================================================

fprintf('=================================================================\n');
fprintf('  HEAD-TO-HEAD COMPARISON: Best Ensemble vs Best FFNN\n');
fprintf('  (Test Set Results)\n');
fprintf('=================================================================\n');
fprintf('%-14s  %-32s  %-10s  %-10s  %-10s  %-8s\n', ...
        'Family', 'Best Model', 'MAE', 'MSE', 'RMSE', 'R²');
fprintf('%s\n', repmat('-', 1, 90));
fprintf('%-14s  %-32s  %-10.7f  %-10.7f  %-10.7f  %-8.4f\n', ...
        'Ensemble', best_ens_name, ...
        best_ens_mae, best_ens_mse, best_ens_rmse, best_ens_r2);
fprintf('%-14s  %-32s  %-10.7f  %-10.7f  %-10.7f  %-8.4f\n', ...
        'FFNN', overall_ffnn_name, ...
        overall_ffnn_mae, overall_ffnn_mse, overall_ffnn_rmse, overall_ffnn_r2);
fprintf('%s\n', repmat('-', 1, 90));

%% --- Determine winner ---
mae_diff_pct = ((best_ens_mae - overall_ffnn_mae) / best_ens_mae) * 100;
if overall_ffnn_mae < best_ens_mae
    fprintf('\n  RESULT: FFNN outperforms Ensemble by %.2f%% on test MAE.\n', ...
            abs(mae_diff_pct));
    winner = 'FFNN';
elseif best_ens_mae < overall_ffnn_mae
    fprintf('\n  RESULT: Ensemble outperforms FFNN by %.2f%% on test MAE.\n', ...
            abs(mae_diff_pct));
    winner = 'Ensemble';
else
    fprintf('\n  RESULT: Both families achieved equal test MAE.\n');
    winner = 'Tie';
end

fprintf('\n');

%% =====================================================================
%% 3. FULL RANKING TABLE — all models from both families together
%% =====================================================================

fprintf('=================================================================\n');
fprintf('  FULL RANKING — All Models by Test MAE (low to high)\n');
fprintf('=================================================================\n');

all_names = [ensemble_names, ...
             strcat(nn_labels(:,1)', {' ('}, nn_labels(:,2)', {')'}) , ...
             wide_labels'];
all_mae   = [ensemble_test_mae, nn_results_test(:,1)', wide_results_test(:,1)'];
all_r2    = [ensemble_test_r2,  nn_results_test(:,4)', wide_results_test(:,4)'];

% Label each model with its family
family_labels = [repmat({'Ensemble'}, 1, 4), ...
                 repmat({'FFNN'},     1, 15), ...
                 repmat({'FFNN-Wide'},1, 3)];

[sorted_mae, sort_idx] = sort(all_mae, 'ascend');
sorted_names   = all_names(sort_idx);
sorted_r2      = all_r2(sort_idx);
sorted_families= family_labels(sort_idx);

fprintf('%-4s  %-14s  %-34s  %-10s  %-8s\n', ...
        'Rank', 'Family', 'Model', 'Test MAE', 'Test R²');
fprintf('%s\n', repmat('-', 1, 78));
for i = 1:numel(sorted_mae)
    fprintf('%-4d  %-14s  %-34s  %-10.7f  %-8.4f\n', ...
            i, sorted_families{i}, sorted_names{i}, sorted_mae(i), sorted_r2(i));
end
fprintf('\n');

%% =====================================================================
%% 4. SAVE COMPARISON RESULTS TO CSV
%% =====================================================================

T_compare = table( ...
    sorted_names', sorted_families', sorted_mae', sorted_r2', ...
    'VariableNames', {'Model', 'Family', 'Test_MAE', 'Test_R2'});
writetable(T_compare, fullfile('tables', 'Table6_full_comparison_ranking.csv'));
fprintf('  Full ranking saved: Table6_full_comparison_ranking.csv\n\n');

%% Save head-to-head summary
T_headtohead = table( ...
    {'Ensemble'; 'FFNN'}, ...
    {best_ens_name; overall_ffnn_name}, ...
    [best_ens_mae;  overall_ffnn_mae], ...
    [best_ens_mse;  overall_ffnn_mse], ...
    [best_ens_rmse; overall_ffnn_rmse], ...
    [best_ens_r2;   overall_ffnn_r2], ...
    'VariableNames', {'Family','Best_Model','Test_MAE','Test_MSE','Test_RMSE','Test_R2'});
writetable(T_headtohead, fullfile('tables', 'Table7_head_to_head.csv'));
fprintf('  Head-to-head summary saved: Table7_head_to_head.csv\n\n');

%% =====================================================================
%% 5. FIGURE — Comparison Bar Chart (MAE by model family)
%% =====================================================================

fig_cmp = figure('Name', 'Ensemble vs FFNN Comparison', ...
                 'Position', [100 100 1100 520], 'Visible', 'off');

%% Prepare grouped bar data
%% Group 1: Ensemble models (4)
%% Group 2: Standard FFNN best per architecture (5 best — one per arch)
%%   Find best activation per architecture
arch_names  = {'Narrow','Medium','Wide','Bi-layer','Tri-layer'};
arch_mae    = zeros(1, 5);
for arch = 1:5
    rows_for_arch = (arch-1)*3 + (1:3);   % each arch has 3 rows (ReLU, Tanh, Sigmoid)
    [arch_mae(arch), ~] = min(nn_results_test(rows_for_arch, 1));
end
%% Group 3: Wide Tri-Layer (3 configs)
wide_mae_vec = wide_results_test(:,1)';

subplot(1,2,1);
%% Left panel: all ensemble models
bar_colors_ens  = [0.70 0.85 1.00; 0.40 0.65 0.90; 0.15 0.45 0.75; 0.05 0.25 0.55];
b = bar(ensemble_test_mae, 'FaceColor', 'flat');
b.CData = bar_colors_ens;
xticks(1:4);
xticklabels({'Linear\nRegression','Decision\nTree','Bagged\nTrees','Boosted\nTrees'});
ylabel('Test MAE (normalised SOC)');
title('Ensemble Methods — Test MAE', 'FontWeight', 'bold');
grid on; box off;
for k = 1:4
    text(k, ensemble_test_mae(k) + max(ensemble_test_mae)*0.02, ...
         sprintf('%.5f', ensemble_test_mae(k)), ...
         'HorizontalAlignment','center','FontSize',7.5);
end

subplot(1,2,2);
%% Right panel: best FFNN per architecture + wide tri-layer best
all_ffnn_plot = [arch_mae, best_wide_mae];
bar_labels    = {'Narrow','Medium','Wide','Bi-layer','Tri-layer','Wide\nTri-Layer\n[best]'};
bar_colors_nn = [0.75 0.92 0.75; 0.50 0.78 0.50; 0.25 0.62 0.25; ...
                 0.10 0.48 0.10; 0.05 0.35 0.05; 0.85 0.65 0.10];
b2 = bar(all_ffnn_plot, 'FaceColor', 'flat');
b2.CData = bar_colors_nn;
xticks(1:6);
xticklabels(bar_labels);
ylabel('Test MAE (normalised SOC)');
title('FFNN Architectures — Best Test MAE per Architecture', 'FontWeight', 'bold');
grid on; box off;
for k = 1:6
    text(k, all_ffnn_plot(k) + max(all_ffnn_plot)*0.02, ...
         sprintf('%.5f', all_ffnn_plot(k)), ...
         'HorizontalAlignment','center','FontSize',7.5);
end

sgtitle(sprintf('Comparison of Ensemble Methods and FFNNs for SOC Estimation\nOverall Winner: %s', winner), ...
        'FontWeight','bold','FontSize',12);

saveas(fig_cmp, fullfile('figures', 'Figure7_ensemble_vs_ffnn_comparison.png'));
fprintf('  Figure 7 saved: Figure7_ensemble_vs_ffnn_comparison.png\n\n');
close(fig_cmp);

%% =====================================================================
%% 6. FIGURE — R² Comparison (how well each family explains SOC variance)
%% =====================================================================

fig_r2 = figure('Name', 'R² Comparison', ...
                'Position', [100 100 700 480], 'Visible', 'off');

r2_values = [best_ens_r2, overall_ffnn_r2];
r2_labels = {sprintf('Best Ensemble\n(%s)', best_ens_name), ...
             sprintf('Best FFNN\n(%s)', strtrim(overall_ffnn_name(1:min(20,end))))};
b3 = bar(r2_values, 0.5, 'FaceColor', 'flat');
b3.CData = [0.30 0.60 0.90; 0.20 0.70 0.30];
xticks(1:2); xticklabels(r2_labels);
ylim([max(0, min(r2_values)-0.05), 1.02]);
ylabel('R² (Test Set)');
title('R² Score: Best Ensemble vs Best FFNN', 'FontWeight', 'bold');
yline(1.0, '--k', 'Perfect fit (R²=1)', 'LabelHorizontalAlignment','left');
grid on; box off;
for k = 1:2
    text(k, r2_values(k) - 0.015, sprintf('R² = %.4f', r2_values(k)), ...
         'HorizontalAlignment','center','FontSize',10,'FontWeight','bold','Color','white');
end

saveas(fig_r2, fullfile('figures', 'Figure8_r2_comparison.png'));
fprintf('  Figure 8 saved: Figure8_r2_comparison.png\n\n');
close(fig_r2);

fprintf('Stage 7 complete — comparison summary done.\n');
fprintf('Winner on test MAE: %s\n\n', winner);