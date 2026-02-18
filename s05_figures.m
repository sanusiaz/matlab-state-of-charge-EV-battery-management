%% =============================================================================
%% s05_figures.m  — Generate all figures from the paper
%%
%% Figure list (matches paper numbering):
%%   Fig 2  — Training data profile (V, I, T, avg V, avg I, SOC)
%%   Fig 4  — MAE & RMSE comparison — ensemble models (training)
%%   Fig 5  — MAE & RMSE comparison — ensemble models (testing)
%%   Fig 10 — RMSE bar chart across all NN models
%%   Fig 11 — Residual plot — wide neural network (best model)
%%   Fig 12 — Residual plot — tri-layered network
%%   Fig 13 — Validation performance curve (single-layer, simulated)
%%   Fig 14 — Training/test regression plot (single-layer)
%%   Fig 15 — Validation performance curve (tri-layer)
%%   Fig 16 — Training/test regression plot (tri-layer)
%%   Fig 17 — Error histogram (single-layer FFNN)
%%   Fig 18 — Error histogram (tri-layered FFNN)
%% =============================================================================

fprintf('--- Stage 5: Generating all figures ---\n');

saveFig = @(name) saveas(gcf, [name, '.png']);

%% =====================================================================
%% FIG 2 — Training data profile (6-panel, matches paper Fig 2)
%% =====================================================================
figure('Name','Fig2 - Training Data Profile','NumberTitle','off',...
       'Position',[100 100 1100 650]);

nPlot  = min(5000, numel(y_train));   % plot a sample to keep it fast
xAxis  = (1:nPlot)';
Vtr    = X_train_raw(1:nPlot, 1);
Itr    = X_train_raw(1:nPlot, 2);
Ttr    = X_train_raw(1:nPlot, 3);
Ytr    = y_train_raw(1:nPlot);

% Normalise each signal 0-1 for the combined display (matches paper)
normIt = @(v) (v - min(v)) ./ (max(v) - min(v) + eps);

subplot(2,3,1);
plot(xAxis, normIt(Vtr), 'b', 'LineWidth', 0.5);
title('(a) Voltage'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.1]); grid on;

subplot(2,3,2);
plot(xAxis, normIt(Itr), 'b', 'LineWidth', 0.5);
title('(b) Current'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.1]); grid on;

subplot(2,3,3);
plot(xAxis, normIt(Ttr), 'b', 'LineWidth', 0.5);
title('(c) Temperature'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.1]); grid on;

subplot(2,3,4);
avgV = movmean(normIt(Vtr), 50);
plot(xAxis, avgV, 'b', 'LineWidth', 0.5);
title('(d) Avg Voltage'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.1]); grid on;

subplot(2,3,5);
avgI = movmean(normIt(Itr), 50);
plot(xAxis, avgI, 'b', 'LineWidth', 0.5);
title('(e) Avg Current'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.1]); grid on;

subplot(2,3,6);
plot(xAxis, normIt(Ytr), 'b', 'LineWidth', 0.5);
title('(f) State of Charge (SOC)'); xlabel('Sample'); ylabel('Normalised'); ylim([-0.1 1.3]); grid on;

sgtitle('Fig 2 — Training Dataset Feature Profile');
saveFig('Fig2_training_data_profile');
fprintf('  Saved: Fig2_training_data_profile.png\n');


%% =====================================================================
%% FIG 4 — MAE & RMSE comparison — ensemble models (TRAINING)
%% =====================================================================
trainMAEs  = [results_train.lr_mae,  results_train.tree_mae, ...
              results_train.bag_mae, results_train.boost_mae];
trainRMSEs = [results_train.lr_rmse, results_train.tree_rmse, ...
              results_train.bag_rmse,results_train.boost_rmse];
modX       = 1:4;
mLabels    = {'Lin Reg','Tree','Bagged','Boosted'};

figure('Name','Fig4 - Ensemble Training','NumberTitle','off',...
       'Position',[100 100 900 380]);
subplot(1,2,1);
scatter(modX, trainMAEs, 80, 'r+', 'LineWidth', 2); hold on;
plot(modX, trainMAEs, 'r--');
set(gca,'XTick',modX,'XTickLabel',mLabels,'XTickLabelRotation',20);
ylabel('MAE'); title('MAE Validation (Training)'); grid on;

subplot(1,2,2);
scatter(modX, trainRMSEs, 80, 'r+', 'LineWidth', 2); hold on;
plot(modX, trainRMSEs, 'r--');
set(gca,'XTick',modX,'XTickLabel',mLabels,'XTickLabelRotation',20);
ylabel('RMSE'); title('RMSE Validation (Training)'); grid on;
sgtitle('Fig 4 — Ensemble Model Training Comparison');
saveFig('Fig4_ensemble_training');
fprintf('  Saved: Fig4_ensemble_training.png\n');


%% =====================================================================
%% FIG 5 — MAE & RMSE comparison — ensemble models (TESTING)
%% =====================================================================
testMAEs  = [results_test.lr_mae,  results_test.tree_mae, ...
             results_test.bag_mae, results_test.boost_mae];
testRMSEs = [results_test.lr_rmse, results_test.tree_rmse, ...
             results_test.bag_rmse,results_test.boost_rmse];

figure('Name','Fig5 - Ensemble Testing','NumberTitle','off',...
       'Position',[100 100 900 380]);
subplot(1,2,1);
scatter(modX, testMAEs, 80, 'r+', 'LineWidth', 2); hold on;
plot(modX, testMAEs, 'r--');
set(gca,'XTick',modX,'XTickLabel',mLabels,'XTickLabelRotation',20);
ylabel('MAE'); title('MAE Test'); grid on;

subplot(1,2,2);
scatter(modX, testRMSEs, 80, 'r+', 'LineWidth', 2); hold on;
plot(modX, testRMSEs, 'r--');
set(gca,'XTick',modX,'XTickLabel',mLabels,'XTickLabelRotation',20);
ylabel('RMSE'); title('RMSE Test'); grid on;
sgtitle('Fig 5 — Ensemble Model Test Comparison');
saveFig('Fig5_ensemble_testing');
fprintf('  Saved: Fig5_ensemble_testing.png\n');


%% =====================================================================
%% FIG 10 — RMSE bar chart: ALL neural network models (test)
%% =====================================================================
allRMSEs  = nn_results_test(:, 3);   % column 3 = RMSE
barLabels = cell(15, 1);
for r = 1:15
    barLabels{r} = sprintf('%s\n%s', nn_labels{r,1}, nn_labels{r,2});
end

figure('Name','Fig10 - All NN RMSE','NumberTitle','off','Position',[100 100 950 420]);
bar(1:15, allRMSEs, 'FaceColor', [0.2 0.4 0.8]);
set(gca,'XTick',1:15,'XTickLabel',barLabels,'XTickLabelRotation',45,'FontSize',7);
ylabel('RMSE'); title('Fig 10 — RMSE Across All Neural Network Models (Test)');
grid on;
% Annotate the minimum
[minVal, minIdx] = min(allRMSEs);
text(minIdx, minVal, sprintf('  Best\n  %.4f', minVal), ...
     'FontSize',8,'Color','red','FontWeight','bold');
saveFig('Fig10_all_nn_rmse');
fprintf('  Saved: Fig10_all_nn_rmse.png\n');


%% =====================================================================
%% FIG 11 — Residual plot: WIDE neural network (test predictions)
%%          Using the wide single-layer [100] model (row 7 = Wide+ReLU)
%% =====================================================================
wide_relu_net = fitrnet(X_train, y_train, ...
    'LayerSizes',1000,'Activations','relu', ...
    'Standardize',false,'IterationLimit',1000,'Lambda',0,'Verbose',0);
fig11_pred  = predict(wide_relu_net, X_test);
fig11_resid = y_test - fig11_pred;

figure('Name','Fig11 - Wide NN Residuals','NumberTitle','off','Position',[100 100 800 400]);
scatter(fig11_pred, fig11_resid, 4, [0.85 0.3 0.1], 'filled', ...
        'MarkerFaceAlpha', 0.3); hold on;
yline(0,'k--','LineWidth',1.2);
xlabel('Predicted Response'); ylabel('Residuals (Var6)');
title('Fig 11 — Residual Plot: Wide Neural Network');
xlim([0 1]); grid on;
text(0.05, max(fig11_resid)*0.85, 'Region of Non-Linearity', ...
     'FontSize', 8, 'Color', 'k');
saveFig('Fig11_residual_wide_nn');
fprintf('  Saved: Fig11_residual_wide_nn.png\n');


%% =====================================================================
%% FIG 12 — Residual plot: Tri-layered FFNN (best model from Stage 4)
%% =====================================================================
fig12_resid = best_true_real - best_pred_real;

figure('Name','Fig12 - Tri-layer Residuals','NumberTitle','off','Position',[100 100 800 400]);
scatter(best_pred_real, fig12_resid, 4, [0.85 0.3 0.1], 'filled', ...
        'MarkerFaceAlpha', 0.3); hold on;
yline(0,'k--','LineWidth',1.2);
xlabel('Predicted Response'); ylabel('Residuals (Var6)');
title('Fig 12 — Residual Plot: Tri-layered FFNN');
grid on;
saveFig('Fig12_residual_trilayer');
fprintf('  Saved: Fig12_residual_trilayer.png\n');


%% =====================================================================
%% FIG 13 — Validation performance curve: Single-layer FFNN
%%          (MSE loss vs iterations — approximate using convergence data)
%% =====================================================================
% fitrnet does not return epoch-by-epoch history, so we simulate the
% convergence curve shape by training at increasing iteration budgets
fprintf('  Generating Fig13 convergence curve (may take ~30 sec)...\n');
iterPts   = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
mse_curve = zeros(size(iterPts));
for k = 1:numel(iterPts)
    tmpNet = fitrnet(X_train, y_train, 'LayerSizes', 10, ...
        'Activations','tanh','Standardize',false, ...
        'IterationLimit',iterPts(k),'Lambda',0,'Verbose',0);
    p = predict(tmpNet, X_test);
    mse_curve(k) = mean((y_test - p).^2);
end

figure('Name','Fig13 - Single-layer Validation','NumberTitle','off','Position',[100 100 700 400]);
semilogy(iterPts, mse_curve, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Iterations'); ylabel('Mean Squared Error (MSE)');
title(sprintf('Fig 13 — Single-Layer FFNN Convergence\nBest MSE: %.5f', min(mse_curve)));
grid on; legend('Validation MSE');
saveFig('Fig13_singlelayer_convergence');
fprintf('  Saved: Fig13_singlelayer_convergence.png\n');


%% =====================================================================
%% FIG 14 — Training/Test regression plot: Single-layer FFNN
%% =====================================================================
sl_net       = fitrnet(X_train, y_train, 'LayerSizes', 10, ...
    'Activations','tanh','Standardize',false, ...
    'IterationLimit',1000,'Lambda',0,'Verbose',0);
sl_pred_tr   = predict(sl_net, X_train);
sl_pred_te   = predict(sl_net, X_test);
R_tr = corr(y_train, sl_pred_tr)^2;
R_te = corr(y_test,  sl_pred_te)^2;

figure('Name','Fig14 - Single-layer Regression','NumberTitle','off','Position',[100 100 900 420]);
subplot(1,2,1);
scatter(y_train, sl_pred_tr, 3, 'k', 'filled','MarkerFaceAlpha',0.2); hold on;
rl = refline(1,0); rl.Color='r'; rl.LineWidth=1.5;
xlabel('Target'); ylabel('Output'); title(sprintf('Training: R=%.5f', sqrt(R_tr)));
legend('Data','Fit','Y=T'); grid on;

subplot(1,2,2);
scatter(y_test, sl_pred_te, 3, 'k','filled','MarkerFaceAlpha',0.2); hold on;
rl = refline(1,0); rl.Color='r'; rl.LineWidth=1.5;
xlabel('Target'); ylabel('Output'); title(sprintf('Test: R=%.5f', sqrt(R_te)));
legend('Data','Fit','Y=T'); grid on;

sgtitle('Fig 14 — Single-Layer FFNN Training & Test Performance');
saveFig('Fig14_singlelayer_regression');
fprintf('  Saved: Fig14_singlelayer_regression.png\n');


%% =====================================================================
%% FIG 15 — Validation convergence: Tri-layered FFNN
%% =====================================================================
fprintf('  Generating Fig15 tri-layer convergence curve...\n');
mse_tri = zeros(size(iterPts));
for k = 1:numel(iterPts)
    tmpNet = fitrnet(X_train, y_train, 'LayerSizes', [10,10,10], ...
        'Activations','relu','Standardize',false, ...
        'IterationLimit',iterPts(k),'Lambda',0,'Verbose',0);
    p = predict(tmpNet, X_test);
    mse_tri(k) = mean((y_test - p).^2);
end

figure('Name','Fig15 - Tri-layer Convergence','NumberTitle','off','Position',[100 100 700 400]);
semilogy(iterPts, mse_tri, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Iterations'); ylabel('Mean Squared Error (MSE)');
title(sprintf('Fig 15 — Tri-Layer FFNN Convergence\nBest MSE: %.2e', min(mse_tri)));
grid on; legend('Validation MSE');
saveFig('Fig15_trilayer_convergence');
fprintf('  Saved: Fig15_trilayer_convergence.png\n');


%% =====================================================================
%% FIG 16 — Training/Test regression plot: Tri-layered FFNN (best model)
%% =====================================================================
R_tr_tri = corr(y_train, best_pred_train)^2;
R_te_tri = corr(y_test,  best_pred_test)^2;

figure('Name','Fig16 - Tri-layer Regression','NumberTitle','off','Position',[100 100 900 420]);
subplot(1,2,1);
scatter(y_train, best_pred_train, 3, 'k','filled','MarkerFaceAlpha',0.2); hold on;
rl = refline(1,0); rl.Color='r'; rl.LineWidth=1.5;
xlabel('Target'); ylabel('Output'); title(sprintf('Training: R=%.5f', sqrt(R_tr_tri)));
legend('Data','Fit','Y=T'); grid on;

subplot(1,2,2);
scatter(y_test, best_pred_test, 3, 'k','filled','MarkerFaceAlpha',0.2); hold on;
rl = refline(1,0); rl.Color='r'; rl.LineWidth=1.5;
xlabel('Target'); ylabel('Output'); title(sprintf('Test: R=%.5f', sqrt(R_te_tri)));
legend('Data','Fit','Y=T'); grid on;

sgtitle('Fig 16 — Tri-Layer FFNN Training & Test Performance');
saveFig('Fig16_trilayer_regression');
fprintf('  Saved: Fig16_trilayer_regression.png\n');


%% =====================================================================
%% FIG 17 — Error Histogram: Single-layer FFNN
%% =====================================================================
err_sl_tr = y_train - sl_pred_tr;
err_sl_te = y_test  - sl_pred_te;

figure('Name','Fig17 - Single-layer Error Histogram','NumberTitle','off','Position',[100 100 700 420]);
histogram(err_sl_tr, 20, 'FaceColor', [0.2 0.6 0.2], 'FaceAlpha', 0.6); hold on;
histogram(err_sl_te, 20, 'FaceColor', [0.8 0.6 0.1], 'FaceAlpha', 0.6);
xline(0, 'r--', 'LineWidth', 1.5);
xlabel('Errors = Targets - Outputs'); ylabel('Instances');
title('Fig 17 — Error Histogram: Single-layer FFNN (20 Bins)');
legend('Training','Test','Zero Error'); grid on;
saveFig('Fig17_error_hist_singlelayer');
fprintf('  Saved: Fig17_error_hist_singlelayer.png\n');


%% =====================================================================
%% FIG 18 — Error Histogram: Tri-layered FFNN (best model)
%% =====================================================================
err_tri_tr = y_train - best_pred_train;
err_tri_te = y_test  - best_pred_test;

figure('Name','Fig18 - Tri-layer Error Histogram','NumberTitle','off','Position',[100 100 700 420]);
histogram(err_tri_tr, 20, 'FaceColor', [0.2 0.6 0.2], 'FaceAlpha', 0.6); hold on;
histogram(err_tri_te, 20, 'FaceColor', [0.8 0.6 0.1], 'FaceAlpha', 0.6);
xline(0, 'r--', 'LineWidth', 1.5);
xlabel('Errors = Targets - Outputs'); ylabel('Instances');
title('Fig 18 — Error Histogram: Tri-layered FFNN (20 Bins)');
legend('Training','Test','Zero Error'); grid on;
saveFig('Fig18_error_hist_trilayer');
fprintf('  Saved: Fig18_error_hist_trilayer.png\n');

fprintf('Stage 5 complete — all figures saved.\n\n');