function s = softmax(x)
    exp_of_x = exp(x);
    s = exp_of_x ./ sum(exp_of_x, 1);
end