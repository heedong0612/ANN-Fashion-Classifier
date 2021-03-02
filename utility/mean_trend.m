function t = mean_trend(vec)
    if(size(vec, 2) == 1)
        t = 0;
    else
        t = (vec(size(vec, 2)) - vec(1)) / ((size(vec, 2) - 1));
    end
end