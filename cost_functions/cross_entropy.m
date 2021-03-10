function loss = cross_entropy(a, t)
    loss = -sum(t .* log(a));
end