function pmat = addNoiseToColumns(pmat, num)
    for i = 1:size(pmat, 2)
        pmat(1:size(pmat, 1), i) = addNoise(pmat(1:size(pmat, 1), i), num);
    end
end