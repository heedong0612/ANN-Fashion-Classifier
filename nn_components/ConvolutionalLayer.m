classdef ConvolutionalLayer < handle
    %CONVOLUTIONALLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_channels
        out_channels
        k_size
        stride
        filters
    end
    
    methods
        function obj = ConvolutionalLayer(in_channels, out_channels, k_size, stride)
            obj.in_channels = in_channels;
            obj.out_channels = out_channels;
            obj.k_size;
            obj.stride;
            obj.filters = normrnd(0, 0.1, out_channels, in_channels, k_size, k_size);
        end
        
        function out = forward(P)
            out = zeros(out_channels, size(P, 2) - obj.k_size + 1, size(P, 3) - obj.k_size + 1)
            for i = 1:out_channels
                out(i, :, :) = imfilter(P, filters(i, :, :, :))
            end
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

