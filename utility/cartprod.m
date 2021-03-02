function [values, indices] = cartprod(varargin)

    num_sets = length(varargin);
    sizes = [];
    
    for i = 1:num_sets
        sizes(i) = length(varargin{i});
    end
    
    all_indices = ones(prod(sizes), num_sets);
    all_values = zeros(prod(sizes), num_sets);
    curr_indices = ones(1, num_sets);
    
    for row = 1:size(all_indices, 1)
        all_indices(row, 1:num_sets) = curr_indices;
        for i = 1:num_sets
            all_values(row, i) = varargin{i}(curr_indices(i));
        end
        disp([row all_values(row, 1:num_sets)]);
        
        % row = row + 1;
        
        for i = 1:length(curr_indices)
            if(curr_indices(i) < sizes(i))
                curr_indices(i) = curr_indices(i) + 1;
                break;
            else
                curr_indices(i) = 1;
            end
        end
    end
    values = all_values;
    indices = all_indices;
end