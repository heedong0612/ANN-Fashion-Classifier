function precision_testing()
    % make all folders fisible to matlab
    addpath('cost_functions');
    addpath('d_cost_functions');
    addpath('transfer_functions');
    addpath('d_transfer_functions');
    addpath('utility');
    addpath('data');
    addpath('nn_components');
    addpath('augmented');
    
    all_submission_data = readmatrix('test.csv'); % read all 10,000 submission datapoints into matrix
    submission_data = all_submission_data(:, 2:785)' * (1/255); % get rid of the useles "id" column in the submission file
    
    % split training and validation data
    all_examples = readmatrix('train.csv');
    all_labels = to_one_hot(all_examples(:, 2), 0, 9);
    all_examples = all_examples(:, 3:786)' * (1/255);
    
    % Training datapoints out of 60,000. The rest are used for validation
    TRAIN_SIZE = 50000;
    valid_data = all_examples(:, (TRAIN_SIZE + 1):60000);
    valid_labels = all_labels(:, (TRAIN_SIZE + 1):60000);
    
    % LOAD MODELS 
    % load mlp models from .mat files
    model_files = [
        "models/HansModal_2021_3_12_22_17_METRIC_904.mat"
        "models/AdalynModelina_2021_3_12_1_15_METRIC_9014.mat"
        "models/KateModelle_aug_2021_3_11_14_36_METRIC_8926.mat"
        "models/aug_model_2021_3_4_20_35.mat"
        "models/aug_model_2021_3_4_5_9.mat"
        "models/model_2021_3_3_5_14.mat"
        "models/aug_model_2021_3_7_3_11.mat"
        "models/model_2021_3_2_14_27.mat"
        "models/model_2021_3_2_17_49.mat"
        "models/model_2021_3_3_5_14.mat"];
    
    precisions = [];
    
    % vector same size as precision vector.  each index stores the name of
    % the model with the highest precision for that respective index (i.e. class)
    best_model_per_class = [];
    
    % removes duplicates from best_model_per_class, only showing unique models
    unique_models = [""];
    
    % compute precisions for each model
    for i = 1:length(model_files)
        load(model_files(i), "mlp");
        preds = hardmax(mlp.frozen_forward(valid_data));
        
        % !!! make sure compute_prec_rec_weight is set to only return precisions
        precisions = [precisions compute_prec_rec_weight(preds, valid_labels)];
    end
    disp(precisions);
    writematrix([model_files round(precisions, 3)'], "all_precisions.txt");
    
    % find models with highest precision in one or more classes
    for i = 1:size(precisions, 1)
        class_precs = precisions(i, 1:size(precisions, 2));
        [maxval, maxi] = max(class_precs);
        class_best = model_files(maxi);
        
        best_model_per_class = [best_model_per_class; class_best];
        if ~ismember(class_best, unique_models)
            unique_models = [unique_models; class_best];
        end
    end
    disp(best_model_per_class);
    disp(unique_models);
end