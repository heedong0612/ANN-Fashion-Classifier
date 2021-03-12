classdef PerceptronLayer < handle
    properties
        trans_func % layer transfer function
        d_trans_func % derivative of layer transfer function
        batch_count % current count of how many examples have been seen in the current batch
        lr_max
        lr_min
        W % layer weights
        b % layer biases
        n % most recent net input, needed for backprop (backward)
        a % most recent activation, needed for last layer sensitivity
        s % most recent sensitivity, needed for backprop
        p % most recent input vector, needed to compute each p's sensitivity (add_to_s)
        avg_s % sum of sensitivities in the current batch, needed for gradient descent (update)
        avg_sp % sum of the sensitivity * p-input' matrices in the current batch. Averaged at update
        last_W_update
        last_b_update
        momentum
    end
    
    methods
        function obj = PerceptronLayer(arg1, arg2, trans_func, d_trans_func, learn_rate_max, learn_rate_min, momentum, std)
            obj.trans_func = trans_func;
            obj.d_trans_func = d_trans_func;
            obj.lr_max = learn_rate_max;
            obj.lr_min = learn_rate_min;
            obj.momentum = momentum;
            
            % If both arguments are scalers, create random weight matrix and
            % bias vector. Otherwise, use the provided weights and biases
            if size(arg1) == [1 1] & size(arg2) == [1 1]
                obj.W = normrnd(0, std, arg1, arg2);
                obj.b = normrnd(0, std, arg1, 1);
            else
                obj.W = arg1;
                obj.b = arg2;
            end
            
            obj.reset_for_next_batch();
            obj.last_W_update = zeros(size(obj.W));
            obj.last_b_update = zeros(size(obj.W, 1), 1);
        end
        
        function output = forward(obj, p)
            obj.batch_count = obj.batch_count + 1;
            obj.p = p;
            
            obj.n = obj.W * p + obj.b;
            obj.a = obj.trans_func(obj.n);
            output = obj.a;
        end
        
        function output = frozen_forward(obj, p)
            output = obj.trans_func(obj.W * p + obj.b);
        end
        
        function backward(obj, next_W, next_s)
            obj.add_to_s(obj.d_trans_func(obj.n) .* (next_W' * next_s));
        end
        
        function update(obj, epoch_progress)
            obj.avg_s = obj.avg_s / obj.batch_count;
            obj.avg_sp = obj.avg_sp / obj.batch_count;
            
            if obj.lr_min == obj.lr_max 
               lr = obj.lr_min; 
            else
               lr = obj.lr_max - (obj.lr_max - obj.lr_min) * epoch_progress;
            end
            
            disp("epoch_ prog: " + epoch_progress);
            disp("lr: " + lr);
            
            obj.W = obj.W + ((lr * (1 - obj.momentum) * obj.avg_sp) + (obj.momentum * obj.last_W_update));
            obj.b = obj.b + ((lr * (1 - obj.momentum) * obj.avg_s) + (obj.momentum * obj.last_b_update));
            
            obj.reset_for_next_batch();
        end
        
        function add_to_s(obj, s_vec)
            obj.avg_s = obj.avg_s + s_vec;
            obj.avg_sp = obj.avg_sp + (s_vec * obj.p');
            obj.s = s_vec;
        end
        
        function reset_for_next_batch(obj)
            obj.last_W_update = obj.avg_sp;
            obj.last_b_update = obj.avg_s;
            
            obj.avg_sp = zeros(size(obj.W));
            obj.avg_s = zeros(size(obj.W, 1), 1);
            
            obj.batch_count = 0;
        end
        
        function print(obj)
            disp('Weights');
            disp(obj.W);
            disp('');
            
            disp('Biases');
            disp(obj.b);
            disp('');
        end
    end
end