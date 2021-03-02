function error = squared_error(a, t)
    e = a - t;
    error = e .* e;
end