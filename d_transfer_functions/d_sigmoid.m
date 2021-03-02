function s = d_sigmoid(x)
    s = sigmoid(x) .* (1 - sigmoid(x));
end