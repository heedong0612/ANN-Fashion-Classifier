function [best_metric, best_params, all_results] = gridsearch(train_data, train_labels, test_data, test_labels, epochs, batches, lrs, momentums, stds, stop_buff, stop_thresh)

    [search_vals, search_indices] = cartprod(batches, lrs, momentums, stds);
    disp(search_indices);
    disp(search_vals);
    
    result_grid = zeros(length(batches), length(lrs), length(momentums), length(stds));
    
    best_metric = 0;
    best_params = zeros(1, size(search_vals, 2));
    
    for i = 1:size(search_vals, 1)
        disp('gridsearch step');
        disp([i size(search_vals, 1)]);
        
        mlp = MultilayerPerceptron(@squared_error, @d_squared_error);
        mlp.add_layer(PerceptronLayer(32, 784, @linear, @d_linear, ...
        search_vals(i, 2), ...
        search_vals(i, 3), ...
        search_vals(i, 4)));
        
        mlp.add_layer(PerceptronLayer(10, 32, @sigmoid, @d_sigmoid, ...
        search_vals(i, 2), ...
        search_vals(i, 3), ...
        search_vals(i, 4)));
        
        [losses, acc, acc_list] = mlp.fit( ...
            train_data, ...
            train_labels, ...
            epochs, ...
            search_vals(i, 1), ...
            stop_buff, ...
            stop_thresh, ...
            test_data, ...
            test_labels, ...
            @accuracy ...
        );
        
        disp(result_grid( ...
            search_indices(i, 1), ...
            search_indices(i, 2), ...
            search_indices(i, 3), ...
            search_indices(i, 4) ...
        ));
        result_grid( ...
            search_indices(i, 1), ...
            search_indices(i, 2), ...
            search_indices(i, 3), ...
            search_indices(i, 4) ...
        ) = acc;
        disp(result_grid( ...
            search_indices(i, 1), ...
            search_indices(i, 2), ...
            search_indices(i, 3), ...
            search_indices(i, 4) ...
        ));
        if(best_metric < acc)
            best_metric = acc;
            best_params = search_vals(i, :);
        end
    end
    
    all_results = result_grid;
end