function demo_lasso_cv()
% demonstration file for SparseGDLibrary.
%
% This file illustrates how to use this library in case of problems with 
% trace norm minimization. 
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on June. 01, 2017

    clc;
    clear;
    close all;
    
    
    %% prepare dataset
    n = 128;
    d = 10;  
    k = 5; 
    noise_level = 0.01;
    [A, b, ~, ~, lambda_max] = generate_lasso_data(n, d, k, noise_level);    


    %% set algorithms and solver
    algorithm = {'FISTA'};

     
    %% initialize
    % define parameters for cross-validation
    num_cv = 10;
    lambda_unit = lambda_max/num_cv;
    lambda_array = 0+lambda_unit:lambda_unit:lambda_max;
    
    % prepare arrays for solutions
    W = zeros(n, num_cv);
    l1_norm = zeros(num_cv,1);    
    aprox_err = zeros(num_cv,1);  
    
    % set options
    options.w_init = zeros(n,1); 
    options.verbose = 1;
    
    %% perform cross-validations
    for i=1:length(lambda_array)
        lambda = lambda_array(i);
        problem = lasso_problem(A, b, lambda);
        
        [W(:,i), infos] = fista(problem, options);
        l1_norm(i) = infos.reg(end);
        aprox_err(i) = infos.cost(end);
    end
    

    %% plot all
    % l1-norm vs. coefficient
    display_graph('l1-norm','coeffs', algorithm, l1_norm, {W}, 'linear');
    % lambda vs. coefficient
    display_graph('lambda','coeffs', algorithm, lambda_array, {W}, 'linear');
    % l1-norm vs. approximation error
    display_graph('l1-norm','aprox_err', algorithm, l1_norm, {aprox_err}, 'linear');    
    
end




