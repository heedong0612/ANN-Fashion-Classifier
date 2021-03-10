function weights = compute_prec_rec_weight(predictions, labels)
    true_pos = sum(predictions & labels, 2);
    precisions = true_pos ./ sum(predictions, 2); % TP / (TP + FP)
    recalls = true_pos ./ sum(labels, 2); % TP / (TP + FN)
    
    weights = precisions .* recalls;
end