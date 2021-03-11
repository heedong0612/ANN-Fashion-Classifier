% IMPORTANT: the string parameter filename should be in this format:
% "somemodel_yyyy_dd_mm_HH_MM_ss_METRIC_"

% this method deletes all other models in the model directory
% that starts with filename except the one with the best accuracy
function cleanup_models(filename)
   files = ls('models/*.mat');           
   
   best_acc = -1;
   
    for n = 1:height(files)
        if contains(files(n,:), filename)
            % parse and mark best accuracy
            acc = str2num(files(n, strlength(filename)+1:strfind(files(n,:), '.mat')-1));
            best_acc = max(best_acc, acc);
        end
    end 
    disp("best accuary: " + best_acc/100 + "%");
     
    if best_acc == -1
       disp("There was a problem deleting the models.") 
    else
    
        disp("============ Delete files ===========");
        for n = 1:height(files)
            if contains(files(n,:), filename)
                acc = str2num(files(n, strlength(filename)+1:strfind(files(n,:), '.mat')-1));
                if acc < best_acc
                    disp(files(n,:));
                    delete("models/" + files(n,:));
                end
            end
        end     
    
        
    end
end