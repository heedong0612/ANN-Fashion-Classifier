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
    aug_data = readmatrix('augmented/augmented_train_2021_03_03_23_56_09.csv'); 
    aug_label = readmatrix('augmented/augmented_label_2021_03_03_23_56_09.csv');
    disp(size(aug_label));
    
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
    load("models/model_2021_3_3_5_14.mat", "mlp");     
    ensemble.add_model(mlp);
    
    load("models/aug_model_2021_3_4_5_9.mat", "mlp");     
    ensemble.add_model(mlp);
    
    load("models/aug_model_2021_3_4_20_35.mat", "mlp");     
    ensemble.add_model(mlp);
    
    
    % predict labels for submission data
    final_preds = one_hot_to_int(hardmax(ensemble.frozen_forward(submission_data, valid_data, valid_labels)), 0, 9);
    
    % format matrix for submission
    submission_matrix = [(60001:70000)' final_preds];
    
    % write submission matrix 
    t = datetime('now');
    submission_timestamp = "data/ensemble_submission_" + year(t) + '_' + month(t) + '_' + day(t) + '_' + hour(t) + '_' + minute(t) + '.csv';
    writematrix(submission_matrix, submission_timestamp);
    
end