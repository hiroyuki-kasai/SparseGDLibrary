function [] = test_background_subtraction()
% This code partially comes from https://github.com/dlaptev/RobustPCA.

    clc;
    clear;
    close all;
    
     
    %% set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        %algorithms = {'Inexact-robustPCA', 'ADMM-robustPCA'}; 
        algorithms = {'ADMM-robustPCA'}; 
    end
    
    
    %% set paramters
    dataset = 'escalator';
    image_display_flag = true;
    
    
    %% prepare dataset
    if strcmp(dataset, 'hall')
        input_movie = importdata('../data/movie/hall1-200.mat');   
        height = 144;
        width = 176;
        X = input_movie';
    elseif strcmp(dataset, 'escalator')
        input_movie = importdata('../data/movie/escalator.mat');   
        height = 130;
        width = 160;
        X = double(input_movie.X');
    else
        return;
    end
    
    
    %% define problem definitions
    lambda = 1/sqrt(max(size(X)));
    lambda = lambda/3;
    mask = logical(zeros(size(X)));
    problem = robust_pca(X, mask, lambda);

    
    %% initialize
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.max_iter = 50;
        options.verbose = 1;  

        switch algorithms{alg_idx}
            case 'ADMM-robustPCA'
                options.mu = 10*lambda/3;
                [w_list{alg_idx}, info_list{alg_idx}] = admm_robust_pca(problem, options);
                
            case 'Inexact-robustPCA'
                [A_hat, E_hat, iter] = inexact_alm_rpca(X, lambda, 1e-7, options.max_iter);
                w_list{alg_idx}.L = A_hat;
                w_list{alg_idx}.S = E_hat;
                
            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
    end
    
    
    fprintf('\n\n');
    
    
    %% plot all
    close all;
    
    % iter vs cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % iter vs residual (X-L-S)
    display_graph('iter','resi_normg', algorithms, w_list, info_list);    
    % iter vs rank(L)
    display_graph('iter','rank', algorithms, w_list, info_list);       


    %% display images
    if image_display_flag
        total_slices = size(X, 1);
        figure;
        cols = length(algorithms);
        rows = 4;
        for i=1:total_slices

            for alg_idx=1:length(algorithms)
                display_images(height, width, rows, cols, alg_idx, i, X, w_list{alg_idx}, algorithms{alg_idx});
            end
            pause(0.1);
        end
    end     
end


function display_images(height, width, rows, cols, test, frame, X, sub_infos, algorithm)

        subplot(rows, cols, 1 + (test-1));
        X_frame = reshape(X(frame,:),[height width]);
        imagesc(X_frame);
        colormap(gray);axis image;axis off;
        title([algorithm, ': f = ', num2str(frame)]); 

        subplot(rows, cols, cols + 1 + (test-1));
        imagesc(reshape(sub_infos.L(frame,:),[height width]));
        colormap(gray);axis image;axis off;
        title('Low-rank image');

        subplot(rows, cols, 2*cols + 1 + (test-1));
        S_frame = reshape(abs(sub_infos.S(frame,:)),[height width]);
        imagesc(S_frame);
        colormap(gray);axis image;axis off;
        title('Residual (sparse) image');
        
        subplot(rows, cols, 3*cols + 1 + (test-1));
        % median filter in space; threshold
        S_frame_median = (medfilt2(S_frame, [5,5]) > 5) .* X_frame;
        imagesc(S_frame_median);
        colormap(gray);axis image;axis off;        
        title('Median filtered residual image');   
        
end




