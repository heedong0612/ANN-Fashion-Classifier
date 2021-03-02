function r = d_relu(x)
    x(x < 0) = 0;
    x(x >= 0) = 1;
    r = x;
end