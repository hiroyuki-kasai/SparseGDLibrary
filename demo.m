function  demo()
% demonstration file for SparseGDLibrary.
%
% This file illustrates how to use this library in case of "lasso" 
% problem. This demonstrates FISTA and ADMM algorithms.
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on Apr. 24, 2017

    clc;
    clear;
    close all;
    

    %% prepare dataset
    n = 1280; 
    d = 100;  
    k = 15; 
    noise_level = 0.01;
    [A, b, x0, lambda] = generate_lasso_data(n, d, k, noise_level);
    
    
    %% define problem definitions
    problem = lasso_problem(A, b, lambda);

    
    %% perform algorithms (FISTA and ADMM)
    options.w_init = zeros(n,1); 
    options.solution = x0;
    options.verbose = true;
    [w_fista, info_fista] = fista(problem, options);
    [w_admm, info_admm] = admm_lasso(problem, options); 

    
    %% plot all
    close all;
    % display iter vs. cost
    display_graph('iter','cost', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm});
    % display iter vs. cost
    display_graph('iter','sol_optimality_gap', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm});    
    % display iter vs. l1-norm
    display_graph('iter','l1-norm', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm}, 'linear');
    % display coeff position vs. coeff amplitude
    display_graph('coeff_pos','coeff_amp', {'FISTA','ADMM','Original (x0)'}, {w_fista, w_admm, x0}, {w_fista, w_admm, x0}, 'linear', 'line-with-mark', 1);  

end


