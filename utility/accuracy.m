function a = accuracy(predictions, labels)
    count = 0;
    label_size = size(labels);
    
    for i = 1:label_size(2)
        if predictions(1:label_size(1), i) == labels(1:label_size(1), i)
            count = count + 1;
        end
    end
    a = count / label_size(2);
end