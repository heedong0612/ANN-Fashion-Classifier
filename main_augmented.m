% !!! IMPORTNAT !!!

% this file saves the model it trained to a directory called 'models'
% please make sure you have a directory 'models' before running this code

function main_augmented()
    % make all folders fisible to matlab
    addpath('cost_functions');
    addpath('d_cost_functions');
    addpath('transfer_functions');
    addpath('d_transfer_functions');
    addpath('utility');
    addpath('data');
    addpath('nn_components');
    addpath('augmented');
    
    % read data -- CHANGE THE FILE TO YOUR AUGMENTED DATASET
    aug_data = readmatrix('augmented/augmented_train_2021_03_10_19_56_04.csv'); 
    aug_label = readmatrix('augmented/augmented_label_2021_03_10_19_56_04.csv');
    disp(size(aug_label));
    
    all_submission_data = readmatrix('test.csv'); % read all 10,000 submission datapoints into matrix
    submission_data = all_submission_data(:, 2:785)' * (1/255); % get rid of the useles "id" column in the submission file
    
    train_data = aug_data' * (1/255); % normalize datapoints
    train_labels = to_one_hot(aug_label, 0, 9); % convert labels (0-9) to one-hot vectors
    
    % split training and validation data
    all_examples = readmatrix('train.csv');
    all_labels = to_one_hot(all_examples(:, 2), 0, 9);
    all_examples = all_examples(:, 3:786)' * (1/255);
    
    
    % Training datapoints out of 60,000. The rest are used for validation
    TRAIN_SIZE = 50000;
    valid_data = all_examples(:, (TRAIN_SIZE + 1):60000);
    valid_labels = all_labels(:, (TRAIN_SIZE + 1):60000);
    
    % hyperparameters
    epochs = 60;
    stop_buff = 1;
    stop_thresh = -1;
    std = 0.4;
    batch_size = 24;
    momentum = 0.9;
    lr = 0.05;
    
    % build model
    mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);
    
    mlp.add_layer(PerceptronLayer(400, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(250, 400, @sigmoid, @d_sigmoid, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(100, 250, @sigmoid, @d_sigmoid, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(10, 100, @softmax, @d_linear, lr, momentum, std));
    
    % change each run to make identifying models easier
    model_name = "FreddieModelson_aug";
    
    t = datetime('now');
    model_timestamp = "models/" + model_name + "_" + year(t) + '_' + month(t) + '_' + day(t) + '_' + hour(t) + '_' + minute(t);
    
    % train the model
    [losses, acc, acc_list] = mlp.fit( ...
        train_data, ...
        train_labels, ...
        epochs, ...
        batch_size, ...
        stop_buff, ...
        stop_thresh, ...
        valid_data, ...
        valid_labels, ...
        @accuracy, ...
        model_timestamp ...
    );
    disp(" ==================== losses ====================" );
    disp(losses);
    disp(" ==================== acc_list ====================" );
    disp(acc_list);
    
    % predict labels for submission data
    final_preds = one_hot_to_int(hardmax(mlp.frozen_forward(submission_data)), 0, 9);
    
    % format matrix for submission
    submission_matrix = [(60001:70000)' final_preds];
    
    % write submission matrix to data/SUBMISSION.csv
    submission_timestamp = "data/" + model_name + "_submission_" + year(t) + '_' + month(t) + '_' + day(t) + '_' + hour(t) + '_' + minute(t) + '.csv';
    writematrix(submission_matrix, submission_timestamp);
    
    % save the model
    % model_timestamp = "models/models/" + model_name + "_" + year(t) + '_' + month(t) + '_' + day(t) + '_' + hour(t) + '_' + minute(t) + '.mat';
    % save(model_timestamp, "mlp");
end
