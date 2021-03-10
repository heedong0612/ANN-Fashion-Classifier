function s = d_softmax(x)
    % softmax(x) is always used as softmax(sigmoid(x)), so we apply the chain rule here:
    % d_softmax(sigmoid(x)) * d_sigmoid(x)
    s = softmax(x) .* (1 - softmax(x));
end