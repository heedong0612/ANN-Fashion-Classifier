classdef Ensemble < handle
    
    properties
        model_list
    end
    
    methods
        function obj = Ensemble()
            obj.model_list = [];
        end
        
        function add_model(obj, model)
            % add model to list
            obj.model_list = [obj.model_list model];
        end
        
        function [losses, best_metrics, all_metrics] = fit(obj, examples, labels, epochs, batch_size, stop_buff, stop_thresh, test_data, test_labels, metric_func)
            % fit all models in model_list one after another. Also, record
            % loss, best_metric, and all_metrics for each model in a
            % matrix, where each row is the result from one model.
            
            losses = [];
            best_metrics = [];
            all_metrics = [];
            
            for i = 1:size(obj.model_list, 2)
                [curr_losses, curr_best_metrics, curr_all_metrics] = obj.model_list(i).fit( ...
                    examples, ...
                    labels, ...
                    epochs, ...
                    batch_size, ...
                    stop_buff, ...
                    stop_thresh, ...
                    test_data, ...
                    test_labels, ...
                    metric_func ...
                );
                losses = [losses; curr_losses];
                best_metrics = [best_metrics; curr_best_metrics];
                all_metrics = [all_metrics; curr_all_metrics];
            end
            
            disp('FINAL INDIVIDUAL METRICS');
            disp(best_metrics);
            disp('FINAL ENSEMBLE METRIC');
            disp(metric_func(hardmax(obj.frozen_forward(test_data, test_data, test_labels)), test_labels));
        end
        
        function weighted_votes = frozen_forward(obj, vec, test_data, test_labels)
            % perform the voting algorithm. The final votes are the average
            % of the weighted sum of each model's predictions, where the weights are the
            % model's (precision * recall) score per-class.
            
            weighted_votes = zeros(size(test_labels));
            
            for i = 1:size(obj.model_list, 2)
                num_layers = size(obj.model_list(i).layers, 2);
                
                % temporrarily swap out the last layer transfer function for softmax
                % temp_handle = obj.model_list(i).layers(num_layers).trans_func;
                % obj.model_list(i).layers(num_layers).trans_func = @softmax;
                
                % get preditioncs
                preds = obj.model_list(i).frozen_forward(vec);
                
                % revert last layer transfer function to original function
                % obj.model_list(i).layers(num_layers).trans_func = temp_handle;
                
                % compute model weights
                confidence_weights = repmat(compute_prec_rec_weight(hardmax(preds), test_labels), 1, size(vec, 2));
                
                % add to the weighted sum
                weighted_votes = weighted_votes + (confidence_weights .* preds);
            end
            
            % average the weighted sum
            weighted_votes = weighted_votes * (1 / size(vec, 2));
        end
    end
end

