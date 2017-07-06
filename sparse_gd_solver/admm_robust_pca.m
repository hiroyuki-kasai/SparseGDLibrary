function [w, infos] = admm_robust_pca(problem, options)
% The alternating direction method of multipliers (ADMM) algorithm for robust PCA problem.
%
% Inputs:
%       problem     function (cost)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SparseGDLibrary.
%
% The pratial codes are from
%       https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html, and 
%       https://github.com/dlaptev/RobustPCA.
%
% Modified by H.Kasai on Apr. 27, 2017


    % set dimensions and samples
    m = problem.m();
    n = problem.n();
    mask = problem.mask();
    lambda = problem.lambda();
    
    % extract options
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end      
    
    if ~isfield(options, 'max_iter')
        max_iter = 100;
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end   
    
    % augmented lagrangian parameter
    if ~isfield(options, 'mu')
        mu = 10 * lambda;
    else
        mu = options.mu;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end 
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end  
    
    % initialise
    iter = 0;
    w.L = zeros(m, n);
    w.S = zeros(m, n);
    w.Y = zeros(m, n); 
    

    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    R = problem.residual(w);
    resi_normg = norm(R, 'fro');
    infos.resi_normg = resi_normg;
    L_rank = rank(w.L);
    infos.rank = L_rank;
    S_card = problem.cardinality(w.S);
    infos.card = S_card;    
    if isfield(problem, 'reg')
        infos.reg = problem.reg(w);   
    end    
    if store_w
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();  
    
    % print info
    if verbose
        fprintf('ADMM robust PCA: Iter = %03d, cost = %.8e, optgap = %.4e, resi_normg = %.4em, rank(L) = %d, Card(S) = %d\n', iter, f_val, optgap, resi_normg, L_rank, S_card);
    end      

    % main loop
    while (optgap > tol_optgap) && (iter < max_iter)      
        
        % update L
        w.L = problem.prox_trace_norm(w, 1/mu);
        
        % update S
        w.S = problem.prox_l1(w, 1/mu);
        
        % upadate augmented lagrangian multiplier with skipping missing values
        Z = problem.residual(w);
        %Z(mask) = 0; % skip missing values
        
        w.Y = w.Y + mu*Z;
        
        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        R = problem.residual(w);
   
        % calculate norm of gradient
        %gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap]; 
        infos.cost = [infos.cost f_val];
        resi_normg = norm(R, 'fro');   
        infos.resi_normg = [infos.resi_normg resi_normg];
        L_rank = rank(w.L);
        infos.rank = [infos.rank L_rank];
        S_card = problem.cardinality(w.S);
        infos.card = [infos.card S_card];
        if isfield(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end        
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('ADMM robust PCA: Iter = %03d, cost = %.8e, optgap = %.4e, resi_normg = %.4em, rank(L) = %d, Card(S) = %d\n', iter, f_val, optgap, resi_normg, L_rank, S_card);
        end        
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);          
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end     
    
end

