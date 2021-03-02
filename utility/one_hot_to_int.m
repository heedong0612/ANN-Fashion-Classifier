function nums = one_hot_to_int(colwise_onehot, min, max)
    nums = colwise_onehot' * (min:max)';
end

