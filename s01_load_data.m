%% =============================================================================
%% s01_load_data.m  — Load a small, targeted subset of the dataset
%%
%% Strategy (keeps runtime fast for a school project):
%%   - Only reads CSVs from the 25degC subfolder
%%   - From those files, keeps rows where |Current| is near 0.75A or 0.10A
%%     (the discharge test currents you specified)
%%   - Caps at MAX_ROWS total rows so training stays under ~30 seconds
%% =============================================================================

fprintf('--- Stage 1: Loading data ---\n');

DATASET_ROOT  = './Dataset_Li-ion';   % <-- adjust if needed
TARGET_SUBDIR = '25degC';             % only this temperature folder
MAX_ROWS      = 50000;                % safety cap — increase if you want more

%% Current bands to keep (discharge means negative current in this dataset)
%% We accept both signs so the filter is safe regardless of sign convention
CURRENT_TARGETS  = [0.75, 0.10];  % amps
CURRENT_BAND     = 0.15;          % ± tolerance around each target

%% ---- Find CSV files ----
folderPath = fullfile(DATASET_ROOT, TARGET_SUBDIR);
if ~isfolder(folderPath)
    error('Folder not found: %s\n Check DATASET_ROOT in s01_load_data.m', folderPath);
end

csvFiles = dir(fullfile(folderPath, '*.csv'));
if isempty(csvFiles)
    error('No CSV files found in: %s', folderPath);
end
fprintf('Found %d CSV files in ''%s''\n', numel(csvFiles), TARGET_SUBDIR);

%% ---- Load and filter each file ----
allFrames = {};
totalRows = 0;

for j = 1:numel(csvFiles)
    if totalRows >= MAX_ROWS
        fprintf('  Row cap (%d) reached — stopping early.\n', MAX_ROWS);
        break;
    end

    filepath = fullfile(folderPath, csvFiles(j).name);
    tbl      = parseSingleCSV(filepath);

    if isempty(tbl)
        fprintf('  Skipped: %s\n', csvFiles(j).name);
        continue;
    end

    %% Keep only rows near target discharge currents
    absI    = abs(tbl.Current);
    keepMask = false(height(tbl), 1);
    for t = 1:numel(CURRENT_TARGETS)
        keepMask = keepMask | (absI >= CURRENT_TARGETS(t) - CURRENT_BAND & ...
                               absI <= CURRENT_TARGETS(t) + CURRENT_BAND);
    end
    tbl = tbl(keepMask, :);

    if height(tbl) == 0
        fprintf('  No matching current rows: %s\n', csvFiles(j).name);
        continue;
    end

    allFrames{end+1} = tbl; %#ok<AGROW>
    totalRows = totalRows + height(tbl);
    fprintf('  Loaded: %-30s  %d rows (running total: %d)\n', ...
            csvFiles(j).name, height(tbl), totalRows);
end

if isempty(allFrames)
    error(['No data loaded. The 25degC CSVs may not contain rows at ' ...
           '0.75A or 0.10A. Try widening CURRENT_BAND in s01_load_data.m']);
end

%% ---- Combine and clean ----
rawData = vertcat(allFrames{:});
rawData = rawData(1:min(MAX_ROWS, height(rawData)), :);  % enforce cap

%% Coerce feature and target columns to numeric
colsNeeded = {'Voltage', 'Current', 'Temperature'};
if ismember('SOC', rawData.Properties.VariableNames)
    targetCol = 'SOC';
elseif ismember('Capacity', rawData.Properties.VariableNames)
    targetCol = 'Capacity';
    fprintf('  Note: ''SOC'' not found — using ''Capacity'' as target.\n');
else
    error('Neither SOC nor Capacity column found in loaded data.');
end
colsNeeded{end+1} = targetCol;

for i = 1:numel(colsNeeded)
    c = colsNeeded{i};
    raw = rawData.(c);
    if iscell(raw)
        rawData.(c) = cellfun(@(v) str2double(string(v)), raw);
    elseif ~isnumeric(raw)
        rawData.(c) = str2double(string(raw));
    end
end

validRows = ~any(isnan([rawData.Voltage, rawData.Current, ...
                         rawData.Temperature, rawData.(targetCol)]), 2);
rawData   = rawData(validRows, :);

fprintf('\nFinal dataset: %d rows  |  Target column: ''%s''\n', ...
        height(rawData), targetCol);

%% ---- Build feature matrix and target vector ----
X_all = [rawData.Voltage, rawData.Current, rawData.Temperature];
y_all = rawData.(targetCol);

%% ---- Scale to [0, 1] ----
Xmin = min(X_all, [], 1);
Xmax = max(X_all, [], 1);
X_scaled = (X_all - Xmin) ./ (Xmax - Xmin + eps);

yMin = min(y_all);
yMax = max(y_all);
y_scaled = (y_all - yMin) ./ (yMax - yMin + eps);

%% ---- 70/30 split (matches the paper) ----
rng(42);
n        = size(X_scaled, 1);
nTest    = round(0.30 * n);
idx      = randperm(n);
testIdx  = idx(1:nTest);
trainIdx = idx(nTest+1:end);

X_train = X_scaled(trainIdx, :);   X_test = X_scaled(testIdx, :);
y_train = y_scaled(trainIdx);      y_test = y_scaled(testIdx);

fprintf('Split — Train: %d rows  |  Test: %d rows\n', ...
        numel(trainIdx), numel(testIdx));

%% ---- Also store unscaled for plots ----
X_train_raw = X_all(trainIdx, :);
X_test_raw  = X_all(testIdx,  :);
y_train_raw = y_all(trainIdx);
y_test_raw  = y_all(testIdx);

fprintf('Stage 1 complete.\n\n');


%% =============================================================================
%% Helper: parse one LG HG2 CSV (same robust parser as soc_estimation.m)
%% =============================================================================
function tbl = parseSingleCSV(filepath)
    tbl = [];
    fid = fopen(filepath, 'r', 'n', 'UTF-8');
    if fid == -1; return; end
    rawLines = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line); rawLines{end+1} = line; end %#ok<AGROW>
    end
    fclose(fid);

    cleanedLines = strtrim(strrep(rawLines, char(0), ''));

    headerIdx = -1;
    for i = 1:numel(cleanedLines)
        parts = strsplit(cleanedLines{i}, ',');
        if contains(cleanedLines{i},'Time Stamp') && ...
           contains(cleanedLines{i},'Voltage')    && ...
           contains(cleanedLines{i},'Current')    && ...
           contains(cleanedLines{i},'Temperature')&& numel(parts) > 10
            headerIdx = i; break;
        end
    end
    if headerIdx == -1; return; end

    headerLine  = cleanedLines{headerIdx};
    nCols       = numel(strsplit(headerLine, ','));
    unitsIdx    = headerIdx + 1;
    dataLines   = cleanedLines(unitsIdx+1:end);
    dataLines   = dataLines(~cellfun(@isempty, dataLines));
    if isempty(dataLines); return; end

    fixedLines = cell(size(dataLines));
    for k = 1:numel(dataLines)
        parts = strsplit(dataLines{k}, ',');
        n = numel(parts);
        if n < nCols; parts(end+1:nCols) = {''}; end
        if n > nCols; parts = parts(1:nCols); end
        fixedLines{k} = strjoin(parts, ',');
    end

    tmpFile = [tempname, '.csv'];
    fid = fopen(tmpFile, 'w');
    fprintf(fid, '%s\n', headerLine);
    for k = 1:numel(fixedLines); fprintf(fid, '%s\n', fixedLines{k}); end
    fclose(fid);

    try
        opts = detectImportOptions(tmpFile, 'VariableNamingRule', 'preserve');
        opts.DataLines = [2, Inf];
        timeVars = {'Time Stamp','TimeStamp','Timestamp'};
        for tc = 1:numel(timeVars)
            if ismember(timeVars{tc}, opts.VariableNames)
                opts = setvaropts(opts, timeVars{tc}, ...
                    'InputFormat','MM/dd/uuuu hh:mm:ss aa'); break;
            end
        end
        wState = warning('off','all');
        tbl = readtable(tmpFile, opts);
        warning(wState);

        varNames = tbl.Properties.VariableNames;
        dropMask = false(1,numel(varNames));
        for v = 1:numel(varNames)
            col = tbl.(varNames{v});
            if isnumeric(col);  dropMask(v) = all(isnan(col));
            elseif iscell(col); dropMask(v) = all(cellfun(@(x) ...
                (ischar(x)&&isempty(strtrim(x)))||(isnumeric(x)&&isnan(x)),col));
            end
        end
        if any(dropMask); tbl = removevars(tbl, varNames(dropMask)); end
    catch
        tbl = [];
    end
    delete(tmpFile);
end