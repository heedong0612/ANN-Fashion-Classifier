function s = sigmoid(x)
    s = (1 + exp(-x)) .^ (-1);
end