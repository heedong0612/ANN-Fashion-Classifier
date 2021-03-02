function prob1()
    addpath('cost_functions/');
    addpath('d_cost_functions');
    addpath('transfer_functions');
    addpath('d_transfer_functions');
    addpath('utility');
    addpath('mnist_data');
    addpath('mlp_components');
    
    
    train_data = loadMNISTImages('mnist_data/train-images-idx3-ubyte');
    train_labels = to_one_hot(loadMNISTLabels('mnist_data/train-labels-idx1-ubyte'), 0, 9);
    test_data = loadMNISTImages('mnist_data/t10k-images-idx3-ubyte');
    test_labels = to_one_hot(loadMNISTLabels('mnist_data/t10k-labels-idx1-ubyte'), 0, 9);
    
    epochs = 200;
    stop_buff = 3;
    stop_thresh = 0;
    std = 0.1;
    batch_size = 32;
    momentum = 0;
    lr = 0.159;
    
%     batches = [1 8 32 128 512 2048];
%     lrs = logspace(-2, 0, 6);
%     momentums = linspace(0, 0.3, 4);
%     momentums = [0];
%     stds = [0.1];
%     
%     [best_metric, best_params, all_results] = gridsearch(train_data, train_labels, test_data, test_labels, epochs, batches, lrs, momentums, stds, stop_buff, stop_thresh);
%     disp('best results');
%     disp(best_metric);
%     disp(best_params);
%     disp(all_results);
    
    mlp = MultilayerPerceptron(@squared_error, @d_squared_error);
    mlp.add_layer(PerceptronLayer(32, 784, @linear, @d_linear, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(10, 32, @sigmoid, @d_sigmoid, lr, momentum, std));
    
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
    
    plot(losses);
    hold on
    plot(acc_list);
    
    title('MSE Loss and Accuracy per Epoch');
    xlabel('Epoch');
    ylabel('Loss/Accuracy');
    xticks(1:length(losses));
    yticks(0:0.1:1.1);
    % set(gms, 'XTick', 1:length(losses), 'XTickLabels', 1:length(losses));
    % set(gms, 'YTick', 0:0.1:1, 'YTickLabels', 0:0.1:1);
    hold off
end


