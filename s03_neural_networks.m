%% =============================================================================
%% s03_neural_networks.m  — All NN variants from the paper
%%
%% Reproduces Tables 2 & 3:
%%   Narrow (10), Medium (25), Wide (100) — single layer
%%   Bi-layered FFNN (10,10)
%%   Tri-layered FFNN (10,10,10)
%%   Each tested with ReLU, Tanh, Sigmoid activations
%% =============================================================================

fprintf('--- Stage 3: Neural Network variants (Tables 2 & 3) ---\n');

activations   = {'relu', 'tanh', 'sigmoid'};
actLabels     = {'ReLU', 'Tanh', 'Sigmoid'};

%% Storage: rows = models x activations
%% Models: Narrow, Medium, Wide, Bi-layer, Tri-layer  (5 models x 3 activations = 15)
nn_results_train = zeros(15, 4);  % MAE MSE RMSE R2
nn_results_test  = zeros(15, 4);
nn_labels        = cell(15, 2);   % {model name, activation}

row = 0;

modelConfigs = {
    'Narrow',   [10],     1000;
    'Medium',   [25],     1000;
    'Wide',     [100],    1000;
    'Bi-layer', [10,10],  1000;
    'Tri-layer',[10,10,10],1000
};

for m = 1:size(modelConfigs, 1)
    mName   = modelConfigs{m,1};
    mLayers = modelConfigs{m,2};
    mIter   = modelConfigs{m,3};

    for a = 1:numel(activations)
        row = row + 1;
        actName = activations{a};
        fprintf('  Training %s (%s)...\n', mName, actLabels{a});

        net = fitrnet(X_train, y_train, ...
            'LayerSizes',     mLayers, ...
            'Activations',    actName, ...
            'Standardize',    false, ...
            'IterationLimit', mIter, ...
            'Lambda',         0, ...
            'Verbose',        0);

        pred_train = predict(net, X_train);
        pred_test  = predict(net, X_test);

        nn_results_train(row,:) = computeNNMetrics(y_train, pred_train);
        nn_results_test(row,:)  = computeNNMetrics(y_test,  pred_test);
        nn_labels{row,1} = mName;
        nn_labels{row,2} = actLabels{a};
    end
end

fprintf('Stage 3 complete.\n\n');


function metrics = computeNNMetrics(y_true, y_pred)
    mae  = mean(abs(y_true - y_pred));
    mse  = mean((y_true - y_pred).^2);
    rmse = sqrt(mse);
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2   = 1 - ss_res / ss_tot;
    metrics = [mae, mse, rmse, r2];
end