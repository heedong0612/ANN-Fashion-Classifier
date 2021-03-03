function error = d_cross_entropy(a, t)
    % error = -(t ./ a) + ((1-t) ./ (1-a));
    error = t - a;
end