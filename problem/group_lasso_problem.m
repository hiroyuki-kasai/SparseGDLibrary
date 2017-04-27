function [Problem] = group_lasso_problem(A, b, group, lambda)
% This file defines the group lasso (least absolute shrinkage and selection operator) problem. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%       group       group id for "n" elements, e.g., [1 1 2 2 3 3 3 4 4].
%       lambda      l1-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + lambda * sum_g ||w_g||_{K_g}).
%
% "w" is the model parameter of size n vector, and ||z||_{K_g} = (z^t K_g z)^{1/2}.
%
% The penalty term is now a sum over l2 norms defined by the positive definite matrices K_j. 
% If each covariate is in its own group and K_g=I, this reduces to the standard lasso, while 
% if there is only a single group and K_1=I, it reduces to ridge regression.
%
%
% This file is part of SGDLibrary, GDLibrary and SparseGDLibrary.
%
% Created by H.Kasai on Apr. 25, 2017


    d = size(A, 2);
    n = size(A, 2);
    
    Problem.name = @() 'group lasso';    
    Problem.dim = @() d;
    Problem.samples = @() n;
    Problem.lambda = @() lambda;
    Problem.A = @() A;    
    Problem.b = @() b;
    Problem.group = @() group;
    num_group = length(unique(group));
    Problem.num_group = @() num_group;
    
    AtA = A'*A;
    Problem.AtA = @() AtA;
    Atb = A'*b;
    %L = max(eig(AtA));
    fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
    L = eigs(A'*A,1);
    fprintf('is L=%f.\n', L);
    Problem.L = @() L;
    
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda);
    end    

    Problem.cost = @cost;
    function f = cost(w)
        reg_val = reg(w);
        f = 1/2 * sum((A * w - b).^2) + lambda * reg_val;
    end

    % calculate group lasso
    Problem.reg = @reg;
    function r = reg(w)
        r = 0;
        for g = 1:num_group
            idx = (group==g);
            w_g = w(idx);
            %r = r + lambda * sqrt(p_g(g))* norm(b_g,2);
            r = r + norm(w_g,2);
        end        
    end

    Problem.residual = @residual;
    function r = residual(w)
        r = - A * w + b;
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        %g = A' * (A * w - b);
        g = AtA * w - Atb;
    end

    Problem.grad = @grad;
    function g = grad(w, indices)
        error('Not implemted yet.'); 
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        h = AtA;       
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        error('Not implemted yet.');
    end

    % for shooting for group lasso
    Problem.shooting_grad = @shooting_grad;
    function g = shooting_grad(w_jminus, idx)
        g = A(:,idx)' * (b - (A * w_jminus));
    end


end

