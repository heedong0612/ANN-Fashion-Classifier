function ensemble_main()
    % make all folders fisible to matlab
    addpath('cost_functions');
    addpath('d_cost_functions');
    addpath('transfer_functions');
    addpath('d_transfer_functions');
    addpath('utility');
    addpath('data');
    addpath('nn_components');
    addpath('augmented');
    
    % READ ALL DATA
    % read data -- CHANGE THE FILE TO YOUR AUGMENTED DATASET
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
    
    ensemble = Ensemble();
    
    % LOAD MODELS 
    % load mlp models from .mat files 
    model_files = [
        "models/HansModal_2021_3_12_22_17_METRIC_904.mat"
        "models/KateModelle_aug_2021_3_11_14_36_METRIC_8926.mat"
        "models/AdalynModelina_2021_3_12_1_15_METRIC_9014.mat"
        "models/EdnaMode_2021_3_16_4_26_METRIC_8992.mat"
        "models/aug_model_2021_3_4_5_9.mat"
        "models/aug_model_2021_3_7_3_11.mat"];
    
    for i = 1:length(model_files)
        load(model_files(i), "mlp");
        ensemble.add_model(mlp);
    end
        
    disp(accuracy(hardmax(ensemble.frozen_forward(valid_data, valid_data, valid_labels)), valid_labels));
    
    % predict labels for submission data
    final_preds = one_hot_to_int(hardmax(ensemble.frozen_forward(submission_data, valid_data, valid_labels)), 0, 9);
    
    % format matrix for submission
    submission_matrix = [(60001:70000)' final_preds];
    
    % write submission matrix 
    t = datetime('now');
    submission_timestamp = "data/ensemble_with_Edna_" + year(t) + '_' + month(t) + '_' + day(t) + '_' + hour(t) + '_' + minute(t) + '.csv';
    writematrix(submission_matrix, submission_timestamp);
    
end