function do_stop = doEarlyStop(ep, avg_errors, early_stop_buff_size, early_stop_threshold)
    % return true to do an early stop IF:
    % the current error is 0
    % OR BOTH the past <buff_size> errors have a standard deviation below 
    %         a threshold AND there have been at least <buff_size> epochs
    if(ep >= early_stop_buff_size)
        disp(std(avg_errors( (size(avg_errors, 2) - early_stop_buff_size + 1):size(avg_errors, 2) )))
    end
    do_stop = avg_errors(size(avg_errors, 2)) == 0 ...
        || (ep >= early_stop_buff_size ...
            && std(avg_errors( (size(avg_errors, 2) - early_stop_buff_size + 1):size(avg_errors, 2) )) <= early_stop_threshold);
end