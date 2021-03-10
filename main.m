function main()
    % make all folders fisible to matlab
    addpath('cost_functions');
    addpath('d_cost_functions');
    addpath('transfer_functions');
    addpath('d_transfer_functions');
    addpath('utility');
    addpath('data');
    addpath('nn_components');
    
    % Training datapoints out of 60,000. The rest are used for validation
    TRAIN_SIZE = 50000;
    
    % READ ALL DATA
    
    all_data = readmatrix('train.csv'); % read all 60,000 labeled datapoints and labels into matrix
    all_submission_data = readmatrix('test.csv'); % read all 10,000 submission datapoints into matrix
    submission_data = all_submission_data(:, 2:785)' * (1/255); % normalize, and get rid of the useles "id" column in the submission file
    
    all_examples = all_data(:, 3:786)' * (1/255); % normalize datapoints
    disp(all_examples(1:20, 1:10));
    all_labels = to_one_hot(all_data(:, 2), 0, 9); % convert labels (0-9) to one-hot vectors
    
    % split training and validation data
    train_data = all_examples(:, 1:TRAIN_SIZE);
    train_labels = all_labels(:, 1:TRAIN_SIZE);
    test_data = all_examples(:, (TRAIN_SIZE + 1):60000);
    test_labels = all_labels(:, (TRAIN_SIZE + 1):60000);
    
    % hyperparameters
    epochs = 1000;
    stop_buff = 1;
    stop_thresh = -1;
    std = 0.4;
    batch_size = 32;
    momentum = 0;
    lr = 0.05;
    
    % build model
    mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);
    
    mlp.add_layer(PerceptronLayer(256, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(10, 256, @softmax, @d_softmax, lr, momentum, std));
    
    % train the model
    [losses, acc, acc_list] = mlp.fit( ...
        train_data, ...
        train_labels, ...
        epochs, ...
        batch_size, ...
        stop_buff, ...
        stop_thresh, ...
        test_data, ...
        test_labels, ...
        @accuracy ...
    );
    disp(losses);
    disp(acc_list);
    
    % predict labels for submission data
    final_preds = one_hot_to_int(hardmax(mlp.frozen_forward(submission_data)), 0, 9);
    
    % format matrix for submission
    submission_matrix = [(60001:70000)' final_preds];
    
    % write submission matrix to data/SUBMISSION.csv
    writematrix(submission_matrix, 'data/SUBMISSION.csv');
end