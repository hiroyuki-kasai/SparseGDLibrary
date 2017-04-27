function [w, infos] = shooting_grouplasso(problem, options)
% Shooting algorithm for Group LASSO problem.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SparseGDLibrary.
%
% References:
%       M. Yuan and Yi Lin,
%       "ReferenceModel selection and estimation in regression with grouped
%       variables,"
%       J. R. Statist. Soc. B, vol.68, Part 1, pp.49-67, 2006.
%
% Created by H.Kasai on Apr. 24, 2017.
% Modified by H.Kasai on Apr. 25, 2017
%
% This code refers below;
%   (1) http://publish.illinois.edu/xiaohuichen/code/group-lasso-shooting/.
%   Lasso with shooting algorithm
%   by Xiaohui Chen (xhchen@illinois.edu).
%
%   (2) http://www.cs.cmu.edu/~gunhee/software.html.
%   by Gunhee Kim (gunhee@{snu.ac.kr or cs.cmu.edu}).


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  
    lambda = problem.lambda();    

    
    % extract options
    if ~isfield(options, 'tol_sol_optgap')
        tol_sol_optgap = 1.0e-12;
    else
        tol_sol_optgap = options.tol_sol_optgap;
    end      
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end      
    
    if ~isfield(options, 'tol_gnorm')
        tol_gnorm = 1.0e-12;
    else
        tol_gnorm = options.tol_gnorm;
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
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end    
    
    if ~isfield(options, 'solution')
        solution = -Inf;
    else
        solution = options.solution;
    end      
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end
    

    % initialise
    iter = 0;
    group = problem.group();
    num_group = problem.num_group();
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    sol_optgap = norm(w - solution);
    infos.sol_optgap = sol_optgap;        
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
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
        fprintf('Shooting (group lasso): Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e, solution optgap = %.4e\n', iter, f_val, gnorm, optgap, sol_optgap);
    end      

    % main loop
    while (optgap > tol_optgap) && (sol_optgap > tol_sol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        

        % See Eq.(2.4) of Yuan & Lin's paper (2006). 
        for j = 1:num_group
            % w_{-j}
            w_jminus = w; 
            idx = (group==j);
            w_jminus(idx) = 0 ;

            % calclulate Sj
            Sj = problem.shooting_grad(w_jminus, idx);
            w_tmp = (1-lambda*sqrt(nnz(idx))/norm(Sj));
            if w_tmp < 0
                w_tmp = 0; 
            end

            % update in Eq.(2.4)
            w(idx) = w_tmp * Sj;
        end  

        % calculate gradient
        grad = problem.full_grad(w);

        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        sol_optgap = norm(w - solution);
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap]; 
        infos.sol_optgap = [infos.sol_optgap sol_optgap];     
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if isfield(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end        
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('Shooting (group lasso): Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e, solution optgap = %.4e\n', iter, f_val, gnorm, optgap, sol_optgap);
        end  
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);          
    elseif sol_optgap < tol_sol_optgap
        fprintf('Solution optimality gap tolerance reached: tol_sol_optgap = %g\n', tol_sol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end
