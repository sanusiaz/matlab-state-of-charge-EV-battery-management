%% =============================================================================
%% s04_wide_trilayer.m  — Wide Tri-layered FFNN (the paper's best model)
%%
%% Reproduces Table 4:
%%   Three configurations tested:
%%     (a) [100,100,100]  1500 iterations
%%     (b) [100,200,100]  2000 iterations   <-- paper's best
%%     (c) [100,100,100]  1000 iterations
%% =============================================================================

fprintf('--- Stage 4: Wide Tri-layered FFNN (Table 4) ---\n');

configs = {
    [100,100,100], 1500, 'ReLU [100:100:100] 1500 iter';
    [100,200,100], 2000, 'ReLU [100:200:100] 2000 iter';
    [100,100,100], 1000, 'ReLU [100:100:100] 1000 iter'
};

wide_results_train = zeros(3, 4);
wide_results_test  = zeros(3, 4);
wide_labels        = cell(3, 1);
wide_nets          = cell(3, 1);

for c = 1:size(configs, 1)
    layers  = configs{c,1};
    iters   = configs{c,2};
    label   = configs{c,3};
    fprintf('  Training %s...\n', label);

    net = fitrnet(X_train, y_train, ...
        'LayerSizes',     layers, ...
        'Activations',    'relu', ...
        'Standardize',    false, ...
        'IterationLimit', iters, ...
        'Lambda',         0, ...
        'Verbose',        0);

    pred_train = predict(net, X_train);
    pred_test  = predict(net, X_test);

    wide_results_train(c,:) = metricsVec(y_train, pred_train);
    wide_results_test(c,:)  = metricsVec(y_test,  pred_test);
    wide_labels{c}          = label;
    wide_nets{c}            = net;
end

%% Best model = config 2 ([100,200,100], 2000 iter) — store for plotting
best_net        = wide_nets{2};
best_pred_test  = predict(best_net, X_test);
best_pred_train = predict(best_net, X_train);

%% Inverse-scale for real-unit evaluation
best_pred_real  = best_pred_test  .* (yMax - yMin) + yMin;
best_true_real  = y_test          .* (yMax - yMin) + yMin;
best_resid      = best_true_real - best_pred_real;

fprintf('Stage 4 complete.\n\n');


function m = metricsVec(y_true, y_pred)
    mae  = mean(abs(y_true - y_pred));
    mse  = mean((y_true - y_pred).^2);
    rmse = sqrt(mse);
    r2   = 1 - sum((y_true-y_pred).^2) / sum((y_true-mean(y_true)).^2);
    m    = [mae, mse, rmse, r2];
end