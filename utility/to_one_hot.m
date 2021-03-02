function one_hot = to_one_hot(colwise_nums, min, max)
    one_hot = (colwise_nums == min:max)';
end

