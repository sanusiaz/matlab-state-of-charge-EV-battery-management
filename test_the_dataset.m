%% =============================================================================
%% SOC Estimation — LG HG2 Li-ion Battery
%% Wide Tri-Layered Feed-Forward Neural Network using fitrnet()
%%
%% Requirements: Statistics and Machine Learning Toolbox (R2021b+)
%% Does NOT require the Deep Learning Toolbox
%% =============================================================================
clc; clear; close all;

%% =============================================================================
%% CONFIGURATION
%% =============================================================================
DATASET_ROOT = 'C:\Users\DELL\Documents\MATLAB\500 level project\first semester\Dataset_Li-ion';
TEMP_SUBDIRS = {'10degC', '25degC', '40degC', 'n10degC', 'n20degC'};


%% =============================================================================
%% STEP 1: LOAD & COMBINE ALL CSV FILES
%% =============================================================================
fprintf('\n=== STEP 1: Loading Data ===\n');
combinedTable = loadAndCombineCSVs(DATASET_ROOT, TEMP_SUBDIRS);
fprintf('Combined dataset: %d rows x %d columns\n', height(combinedTable), width(combinedTable));


%% =============================================================================
%% STEP 2: FEATURE SELECTION & CLEANING
%% =============================================================================
fprintf('\n=== STEP 2: Preparing Features ===\n');
[X, y, yColName] = prepareFeatures(combinedTable);


%% =============================================================================
%% STEP 3: SCALING (0 to 1) & TRAIN/TEST SPLIT (80/20)
%% =============================================================================
fprintf('\n=== STEP 3: Scaling & Splitting ===\n');
[X_train, X_test, y_train, y_test, Xmin, Xmax, yMin, yMax] = scaleAndSplit(X, y, 0.2);
fprintf('Train size: %d  |  Test size: %d\n', size(X_train,1), size(X_test,1));


%% =============================================================================
%% STEP 4 & 5: BUILD AND TRAIN WIDE TRI-LAYERED FFNN via fitrnet()
%% =============================================================================
fprintf('\n=== STEP 4-5: Building & Training Model ===\n');

net = fitrnet(X_train, y_train, ...
    'LayerSizes',     [64, 64, 64], ...
    'Activations',    'relu', ...
    'Standardize',    false, ...
    'IterationLimit', 1000, ...
    'Lambda',         0, ...
    'Verbose',        1);

fprintf('Training complete.\n');


%% =============================================================================
%% STEP 6: EVALUATE ON TEST SET
%% =============================================================================
fprintf('\n=== STEP 6: Evaluation ===\n');

y_pred_scaled = predict(net, X_test);

y_pred = y_pred_scaled .* (yMax - yMin) + yMin;
y_true = y_test        .* (yMax - yMin) + yMin;

mae_val  = mean(abs(y_true - y_pred));
mse_val  = mean((y_true - y_pred).^2);
rmse_val = sqrt(mse_val);
ss_res   = sum((y_true - y_pred).^2);
ss_tot   = sum((y_true - mean(y_true)).^2);
r2_val   = 1 - (ss_res / ss_tot);

fprintf('\n--- Test Results ---\n');
fprintf('  MAE  : %.4f\n',  mae_val);
fprintf('  MSE  : %.4e\n',  mse_val);
fprintf('  RMSE : %.4f\n',  rmse_val);
fprintf('  R2   : %.4f\n',  r2_val);


%% =============================================================================
%% STEP 7: PLOTS
%% =============================================================================
plotPredictionsVsActual(y_true, y_pred, yColName);
plotResiduals(y_true, y_pred);
showSamplePredictions(X_test, y_pred_scaled, 5);


%% =============================================================================
%% LOCAL FUNCTIONS
%% =============================================================================

function combinedTable = loadAndCombineCSVs(datasetRoot, subdirs)
    allTables = {};

    for i = 1:length(subdirs)
        subdir     = subdirs{i};
        folderPath = fullfile(datasetRoot, subdir);

        if ~isfolder(folderPath)
            fprintf('  Warning: Subfolder not found, skipping: %s\n', folderPath);
            continue;
        end

        csvFiles = dir(fullfile(folderPath, '*.csv'));

        if isempty(csvFiles)
            fprintf('  Warning: No CSV files in: %s\n', folderPath);
            continue;
        end

        fprintf('\nLoading from ''%s'' (%d file(s))...\n', subdir, length(csvFiles));

        for j = 1:length(csvFiles)
            filepath = fullfile(folderPath, csvFiles(j).name);
            tbl = parseLgHg2CSV(filepath);

            if ~isempty(tbl)
                tbl.source_temp = repmat({subdir}, height(tbl), 1);
                allTables{end+1} = tbl; %#ok<AGROW>
                fprintf('  Loaded: %s — %d rows\n', csvFiles(j).name, height(tbl));
            else
                fprintf('  Skipped (no usable data): %s\n', csvFiles(j).name);
            end
        end
    end

    if isempty(allTables)
        error('No data loaded. Check DATASET_ROOT and that CSV files exist in the temperature subfolders.');
    end

    combinedTable = vertcat(allTables{:});
end


function tbl = parseLgHg2CSV(filepath)
    % Robustly parses one LG HG2 CSV:
    %   - Strips null characters
    %   - Dynamically finds the real header row
    %   - Skips the units row immediately after the header
    %   - Pads or trims every data row to match the header column count
    %     (fixes "Arrays have incompatible sizes" from ragged rows)
    %   - Preserves original column names including spaces
    %     (fixes MATLAB auto-renaming Voltage/Current/Temperature)

    tbl = [];

    % ---- Read raw lines ----
    fid = fopen(filepath, 'r', 'n', 'UTF-8');
    if fid == -1
        fprintf('  Warning: Cannot open: %s\n', filepath);
        return;
    end
    rawLines = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            rawLines{end+1} = line; %#ok<AGROW>
        end
    end
    fclose(fid);

    % ---- Clean null characters and whitespace ----
    cleanedLines = strtrim(strrep(rawLines, char(0), ''));

    % ---- Locate header row dynamically ----
    headerIdx = -1;
    for i = 1:length(cleanedLines)
        line  = cleanedLines{i};
        parts = strsplit(line, ',');
        if contains(line, 'Time Stamp')  && ...
           contains(line, 'Voltage')     && ...
           contains(line, 'Current')     && ...
           contains(line, 'Temperature') && ...
           numel(parts) > 10
            headerIdx = i;
            break;
        end
    end

    if headerIdx == -1
        [~, fname, ext] = fileparts(filepath);
        fprintf('  Warning: No header found in %s%s. Skipping.\n', fname, ext);
        return;
    end

    % ---- Determine expected column count from header ----
    headerLine  = cleanedLines{headerIdx};
    headerParts = strsplit(headerLine, ',');
    nCols       = numel(headerParts);

    % ---- Skip units row (immediately after header) ----
    unitsIdx  = headerIdx + 1;
    dataLines = cleanedLines(unitsIdx+1:end);

    % ---- Remove blank lines ----
    dataLines = dataLines(~cellfun(@isempty, dataLines));

    if isempty(dataLines)
        return;
    end

    % ---- FIX: Pad or trim every data row to exactly nCols columns ----
    % This prevents "Arrays have incompatible sizes" from ragged rows.
    fixedDataLines = cell(size(dataLines));
    for k = 1:numel(dataLines)
        parts = strsplit(dataLines{k}, ',');
        n     = numel(parts);
        if n < nCols
            % Pad with empty fields
            parts(end+1:nCols) = {''};
        elseif n > nCols
            % Trim extra trailing fields
            parts = parts(1:nCols);
        end
        fixedDataLines{k} = strjoin(parts, ',');
    end

    % ---- Write to temp file ----
    tmpFile = [tempname, '.csv'];
    fid = fopen(tmpFile, 'w');
    fprintf(fid, '%s\n', headerLine);         % header row
    for k = 1:numel(fixedDataLines)
        fprintf(fid, '%s\n', fixedDataLines{k});
    end
    fclose(fid);

    % ---- Read with readtable, preserving original column names ----
    try
        opts = detectImportOptions(tmpFile, ...
            'VariableNamingRule', 'preserve');  % keeps 'Time Stamp', 'Voltage' etc. as-is
        opts.DataLines = [2, Inf];

        % Fix: explicitly set the datetime format for 'Time Stamp' column
        % to suppress the MM/dd vs dd/MM ambiguity warning
        timeColCandidates = {'Time Stamp', 'TimeStamp', 'Timestamp'};
        for tc = 1:numel(timeColCandidates)
            if ismember(timeColCandidates{tc}, opts.VariableNames)
                opts = setvaropts(opts, timeColCandidates{tc}, ...
                    'InputFormat', 'MM/dd/uuuu hh:mm:ss aa');
                break;
            end
        end

        % Suppress remaining cosmetic warnings
        warnState = warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
        warning('off',  'MATLAB:readtable:AllNonNumericData');
        tbl = readtable(tmpFile, opts);
        warning(warnState);

        % Drop entirely-missing trailing columns
        varNames = tbl.Properties.VariableNames;
        dropMask = false(1, numel(varNames));
        for v = 1:numel(varNames)
            col = tbl.(varNames{v});
            if isnumeric(col) || islogical(col)
                dropMask(v) = all(isnan(col));
            elseif iscell(col)
                dropMask(v) = all(cellfun(@(x) ...
                    (ischar(x) && isempty(strtrim(x))) || ...
                    (isnumeric(x) && isnan(x)), col));
            end
        end
        if any(dropMask)
            tbl = removevars(tbl, varNames(dropMask));
        end

    catch ME
        [~, fname, ext] = fileparts(filepath);
        fprintf('  Warning: Failed to parse %s%s: %s\n', fname, ext, ME.message);
        tbl = [];
    end

    delete(tmpFile);
end


function [X, y, yColName] = prepareFeatures(tbl)
    requiredFeatures = {'Voltage', 'Current', 'Temperature'};
    varNames = tbl.Properties.VariableNames;

    for i = 1:length(requiredFeatures)
        if ~ismember(requiredFeatures{i}, varNames)
            error('Missing required column: ''%s''.\nAvailable columns: %s', ...
                  requiredFeatures{i}, strjoin(varNames, ', '));
        end
    end

    if ismember('SOC', varNames)
        yColName = 'SOC';
    elseif ismember('Capacity', varNames)
        fprintf('Warning: ''SOC'' not found — using ''Capacity'' as target.\n');
        yColName = 'Capacity';
    else
        error('Neither ''SOC'' nor ''Capacity'' found.\nAvailable columns: %s', ...
              strjoin(varNames, ', '));
    end

    % Coerce all relevant columns to numeric
    allCols = [requiredFeatures, {yColName}];
    for i = 1:length(allCols)
        col = allCols{i};
        raw = tbl.(col);
        if iscell(raw)
            tbl.(col) = cellfun(@(v) str2double(string(v)), raw);
        elseif ~isnumeric(raw)
            tbl.(col) = str2double(string(raw));
        end
    end

    % Drop rows with any NaN in the relevant columns
    colData   = [tbl.Voltage, tbl.Current, tbl.Temperature, tbl.(yColName)];
    validRows = ~any(isnan(colData), 2);
    tbl       = tbl(validRows, :);

    if height(tbl) == 0
        error('No valid rows remain after cleaning. Check your CSV contents.');
    end

    fprintf('Target column   : ''%s''\n', yColName);
    fprintf('Rows after clean: %d\n', height(tbl));

    X = [tbl.Voltage, tbl.Current, tbl.Temperature];
    y = tbl.(yColName);
end


function [X_train, X_test, y_train, y_test, Xmin, Xmax, yMin, yMax] = ...
         scaleAndSplit(X, y, testFraction)

    Xmin     = min(X, [], 1);
    Xmax     = max(X, [], 1);
    X_scaled = (X - Xmin) ./ (Xmax - Xmin + eps);

    yMin     = min(y);
    yMax     = max(y);
    y_scaled = (y - yMin) ./ (yMax - yMin + eps);

    rng(42);
    n        = size(X_scaled, 1);
    nTest    = round(testFraction * n);
    idx      = randperm(n);
    testIdx  = idx(1:nTest);
    trainIdx = idx(nTest+1:end);

    X_train = X_scaled(trainIdx, :);
    X_test  = X_scaled(testIdx,  :);
    y_train = y_scaled(trainIdx);
    y_test  = y_scaled(testIdx);
end


function plotPredictionsVsActual(y_true, y_pred, yColName)
    figure('Name', 'Predictions vs Actual', 'NumberTitle', 'off');
    scatter(y_true, y_pred, 5, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
    refLine = linspace(min(y_true), max(y_true), 100);
    plot(refLine, refLine, 'r--', 'LineWidth', 1.5);
    xlabel(['Actual ', yColName]);
    ylabel(['Predicted ', yColName]);
    title('Predicted vs Actual SOC');
    legend('Predictions', 'Perfect fit', 'Location', 'northwest');
    grid on;
    saveas(gcf, 'predictions_vs_actual.png');
    fprintf('Plot saved: predictions_vs_actual.png\n');
end


function plotResiduals(y_true, y_pred)
    residuals = y_true - y_pred;
    figure('Name', 'Residuals', 'NumberTitle', 'off');
    scatter(y_pred, residuals, 5, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
    yline(0, 'r--', 'LineWidth', 1.5);
    xlabel('Predicted SOC');
    ylabel('Residual (Actual - Predicted)');
    title('Residual Plot');
    grid on;
    saveas(gcf, 'residuals.png');
    fprintf('Plot saved: residuals.png\n');
end


function showSamplePredictions(X_test_scaled, y_pred_scaled, n)
    fprintf('\nFirst %d Scaled Predictions:\n', n);
    fprintf('%-12s %-12s %-12s %-20s\n', ...
            'V_scaled', 'I_scaled', 'T_scaled', 'Predicted_SOC_scaled');
    fprintf('%s\n', repmat('-', 1, 58));
    for i = 1:min(n, size(X_test_scaled, 1))
        fprintf('%-12.4f %-12.4f %-12.4f %-20.4f\n', ...
                X_test_scaled(i,1), X_test_scaled(i,2), ...
                X_test_scaled(i,3), y_pred_scaled(i));
    end
end