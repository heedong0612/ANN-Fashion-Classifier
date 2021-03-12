classdef MultilayerPerceptron < handle
    
    properties
        layers
        cost_func
        d_cost_func
    end
    
    methods
        function obj = MultilayerPerceptron(cost_func, d_cost_func)
            obj.layers = [];
            obj.cost_func = cost_func;
            obj.d_cost_func = d_cost_func;
        end
        
        function add_layer(obj, layer)
            obj.layers = [obj.layers layer];
        end
        
        function [losses, best_metric, all_metrics] = fit(obj, examples, labels, epochs, batch_size, stop_buff, stop_thresh, test_data, test_labels, metric_func, checkpoint_name)
            for i = 1:size(obj.layers, 2)
                disp(size(obj.layers(i).W));
            end
            
            avg_losses = [];
            
            batch_losses = zeros(batch_size);
            epoch_losses = [];
            temp_avg_losses = [];
            metric = [];
            best_metric = 0
            disp([0.0000 metric_func(hardmax(obj.frozen_forward(test_data)), test_labels)]);
            
            for e = 1:epochs
                perm = randperm(size(examples, 2));
                examples = examples(:, perm);
                labels = labels(:, perm);
                
                for b = 1:int32(ceil(size(examples, 2) / batch_size) + 1)
                    % fprintf('batch %d / %d\n', b, int32(ceil(size(examples, 2) / batch_size) + 1));
                    
                    start_i = (b - 1) * (batch_size) + 1;
                    end_i = min(b * batch_size, size(examples, 2));
                    if start_i > end_i
                        break
                    end
                    for i = start_i:end_i
                        a = obj.forward(examples(1:size(examples, 1), i));
                        batch_losses(i - start_i + 1) = obj.backward(a, labels(1:size(labels, 1), i));
                    end
                    obj.update();
                    avg_losses = [avg_losses mean(batch_losses(1:(end_i - start_i + 1)))];
                end
                
                metric = [metric metric_func(hardmax(obj.frozen_forward(test_data)), test_labels)];
                disp([e metric(size(metric, 2))]);
                
                if(metric(length(metric)) > best_metric)
                    best_metric = metric(length(metric));
                    obj.try_save_checkpoint(checkpoint_name, best_metric);
                end
                
                temp_avg_losses = avg_losses(:,:);
                epoch_losses = [epoch_losses mean(temp_avg_losses)];
                if(doEarlyStop(e, metric, stop_buff, stop_thresh))
                    break
                end
            end
            
            losses = epoch_losses;
            all_metrics = metric;
        end
        
        function try_save_checkpoint(obj, filename, metric)
            
            % save current model
            metric_str = num2str(metric);
            model_timestamp = filename + "_METRIC_" + metric_str(3:size(metric_str, 2)) + '.mat';
            mlp = obj;
            save(model_timestamp, "mlp");
            
            % find and delete the other model
            files = ls('models/*.mat');
            disp(files);
            filename = extractAfter(filename, "models/") + "_METRIC_";

            for n = 1:height(files)
                
                if contains(files(n,:), filename) & (strtrim(files(n,:)) ~= extractAfter(model_timestamp, "models/")) % to avoid deleting the current model
                   delete("models/" + files(n,:));
                   break
                end
            end 
        end
        
        function out = forward(obj, vec)
            for layer = obj.layers
                vec = layer.forward(vec);
            end
            out = vec;
        end
        
        function out = frozen_forward(obj, vec)
            for layer = obj.layers
                vec = layer.frozen_forward(vec);
            end
            out = vec;
        end
        
        function loss = backward(obj, last_a, target)
            last_n = obj.layers(size(obj.layers, 2)).n;
            last_s = obj.layers(size(obj.layers, 2)).d_trans_func(last_n) .* obj.d_cost_func(last_a, target);
            
            obj.layers(size(obj.layers, 2)).add_to_s(last_s);
            
            for i = (size(obj.layers, 2) - 1):-1:1
                obj.layers(i).backward(obj.layers(i+1).W, obj.layers(i+1).s)
            end
            loss = mean(obj.cost_func(last_a, target));
        end
        
        function update(obj)
            for i = 1:size(obj.layers)
                obj.layers(i).update();
            end
        end
        
        function save_checkpoint(obj, ep)
            filename = ['CP_' num2str(size(obj.layers(0).W, 1))];
            for layer = obj.layers
                filename = [filename '_' num2str(size(layer.W, 2))];
            end
            
            
        end
        
        function load_checkpoint()
            
        end
    end
end

