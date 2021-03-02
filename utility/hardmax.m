function out = hardmax(x)
    for i = 1:size(x, 2)
        [max_num, max_index] = max(x(1:size(x, 1), i));
        x(1:size(x, 1), i) = zeros(size(x, 1), 1);
        x(max_index, i) = 1;
    end
    out = x;
end