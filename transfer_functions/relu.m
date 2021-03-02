function r = relu(x)
    x(x < 0) = 0;
    r = x;
end