function t = my_tanh(x)
    t = (exp(x) - exp(-x)) .* ((exp(x) + exp(-x)) .^ (-1));
end