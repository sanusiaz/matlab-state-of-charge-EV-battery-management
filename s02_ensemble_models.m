%% =============================================================================
%% s02_ensemble_models.m  — Linear Regression, Tree & Ensemble models
%%
%% Reproduces Table 1 from the paper:
%%   Linear Regression, Decision Tree, Bagged Trees, Boosted Trees
%%   Metrics: MAE, MSE, RMSE, R-squared  (training and testing)
%% =============================================================================

fprintf('--- Stage 2: Ensemble models ---\n');

results_train = struct();
results_test  = struct();

%% ---- 1. Linear Regression ----
fprintf('  Training Linear Regression...\n');
mdl_lr        = fitlm(X_train, y_train);
pred_lr_train = predict(mdl_lr, X_train);
pred_lr_test  = predict(mdl_lr, X_test);
[results_train.lr_mae, results_train.lr_mse, results_train.lr_rmse, results_train.lr_r2] = ...
    computeMetrics(y_train, pred_lr_train);
[results_test.lr_mae,  results_test.lr_mse,  results_test.lr_rmse,  results_test.lr_r2] = ...
    computeMetrics(y_test,  pred_lr_test);

%% ---- 2. Decision Tree ----
fprintf('  Training Decision Tree...\n');
mdl_tree        = fitrtree(X_train, y_train, 'MinLeafSize', 8);
pred_tree_train = predict(mdl_tree, X_train);
pred_tree_test  = predict(mdl_tree, X_test);
[results_train.tree_mae, results_train.tree_mse, results_train.tree_rmse, results_train.tree_r2] = ...
    computeMetrics(y_train, pred_tree_train);
[results_test.tree_mae,  results_test.tree_mse,  results_test.tree_rmse,  results_test.tree_r2] = ...
    computeMetrics(y_test,  pred_tree_test);

%% ---- 3. Bagged Ensemble ----
fprintf('  Training Bagged Ensemble Trees (30 learners)...\n');
mdl_bag        = fitrensemble(X_train, y_train, ...
    'Method',            'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners',          templateTree('MinLeafSize', 8));
pred_bag_train = predict(mdl_bag, X_train);
pred_bag_test  = predict(mdl_bag, X_test);
[results_train.bag_mae, results_train.bag_mse, results_train.bag_rmse, results_train.bag_r2] = ...
    computeMetrics(y_train, pred_bag_train);
[results_test.bag_mae,  results_test.bag_mse,  results_test.bag_rmse,  results_test.bag_r2] = ...
    computeMetrics(y_test,  pred_bag_test);

%% ---- 4. Boosted Ensemble ----
fprintf('  Training Boosted Ensemble Trees (30 learners, lr=0.01)...\n');
mdl_boost        = fitrensemble(X_train, y_train, ...
    'Method',            'LSBoost', ...
    'NumLearningCycles', 30, ...
    'LearnRate',         0.01, ...
    'Learners',          templateTree('MinLeafSize', 8));
pred_boost_train = predict(mdl_boost, X_train);
pred_boost_test  = predict(mdl_boost, X_test);
[results_train.boost_mae, results_train.boost_mse, results_train.boost_rmse, results_train.boost_r2] = ...
    computeMetrics(y_train, pred_boost_train);
[results_test.boost_mae,  results_test.boost_mse,  results_test.boost_rmse,  results_test.boost_r2] = ...
    computeMetrics(y_test,  pred_boost_test);

%% ---- Store predictions for later plotting stages ----
ensemble_preds_test = struct(...
    'lr',    pred_lr_test, ...
    'tree',  pred_tree_test, ...
    'bag',   pred_bag_test, ...
    'boost', pred_boost_test);

fprintf('Stage 2 complete.\n\n');

%% =============================================================================
%% LOCAL FUNCTION — must be at the bottom of the script
%% =============================================================================
function [mae, mse, rmse, r2] = computeMetrics(y_true, y_pred)
    mae    = mean(abs(y_true - y_pred));
    mse    = mean((y_true - y_pred).^2);
    rmse   = sqrt(mse);
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2     = 1 - ss_res / ss_tot;
end