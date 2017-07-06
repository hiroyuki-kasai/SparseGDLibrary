function [] = test_image_inpainting()
% This code partially comes from https://github.com/dlaptev/RobustPCA.

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'ADMM-robustPCA'}; 
    end    
    
    
    %% prepare dataset
    % read image and add the mask
    img = double(imread('peppers.png'))/255;
    img = img(41:40+256, 21:20+256);
    msk = zeros(size(img));
    msk(65:192,65:192) = imresize(imread('text.png'), 0.5);
    img_corrupted = img;
    img_corrupted(msk > 0) = nan;
    
    % create a matrix X from overlapping patches
    ws = 16; % window size
    no_patches = size(img, 1) / ws;
    X = zeros(no_patches^2, ws^2);
    k = 1;
    for i = (1:no_patches*2-1)
        for j = (1:no_patches*2-1)
            r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
            r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
            patch = img_corrupted(r1, r2);
            X(k,:) = patch(:);
            k = k + 1;
        end
    end    
    
    
    %% define problem definitions
    lambda = 0.02;
    mask = isnan(X);
    problem = robust_pca(X, mask, lambda);

    
    %% initialize
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.max_iter = 100;
        options.verbose = 1;  

        switch algorithms{alg_idx}
            case {'ADMM-robustPCA'}
                
                options.mu = 1.0;
                [w_list{alg_idx}, info_list{alg_idx}] = admm_robust_pca(problem, options);
                
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
    
    % display iter vs cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display iter vs residual (X-L-S)
    display_graph('iter','resi_normg', algorithms, w_list, info_list);    


    %% display images
    for alg_idx=1:length(algorithms)
        % reconstruct the image from the overlapping patches in matrix L
        img_reconstructed = zeros(size(img));
        img_noise = zeros(size(img));
        k = 1;
        L = w_list{alg_idx}.L;
        S = w_list{alg_idx}.S;
        for i = (1:no_patches*2-1)
            for j = (1:no_patches*2-1)
                % average patches to get the image back from L and S
                % todo: in the borders less than 4 patches are averaged
                patch = reshape(L(k,:), ws, ws);
                r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
                r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
                img_reconstructed(r1, r2) = img_reconstructed(r1, r2) + 0.25*patch;
                patch = reshape(S(k,:), ws, ws);
                img_noise(r1, r2) = img_noise(r1, r2) + 0.25*patch;
                k = k + 1;
            end
        end
        img_final = img_reconstructed;
        img_final(~isnan(img_corrupted)) = img_corrupted(~isnan(img_corrupted));

        % show the results
        figure;
        subplot(2,3,1), imshow(img), title('Original clean image')
        subplot(2,3,2), imshow(msk), title('Msk text image')
        subplot(2,3,3), imshow(img_corrupted), title('Corrupted image')
        subplot(2,3,4), imshow(img_final), title('Recovered image')
        subplot(2,3,5), imshow(img_reconstructed), title('Recovered low-rank')
        subplot(2,3,6), imshow(img_noise), title('Recovered sparse')
    end
    
    
end




