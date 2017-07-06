function [Problem] = trace_norm_convex_clustering(X, L, W, lambda)
% This file defines the TV (Total Variation) problem with L1 norm. 
%
% Inputs:
%       X           data matrix of size dxn.
%       L           difference matrix of size n(n-1)/2 * d.
%       W           similarity matrix for L of size n(n-1)/2 * d.
%       lambda      regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/2 * || X - w ||_2^2 + lambda * w_{i,j} || w_i - w_j ||_* ).
%
% "w" is the centroid matrix of size dxn.
%
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on June 30, 2017


    d = size(X, 2);
    n = size(X, 1);
    
    Problem.name = @() 'convex clustering with trace norm';    
    Problem.dim = @() d;
    Problem.samples = @() n;
    Problem.X = @() X;    
     
    Problem.lambda = @() lambda;     
    
    repW = repmat(W, [1,n]);
    L = repW .* L;
    
    Problem.L = @() L;
      
    Problem.prox = @trace_norm;
    function v = trace_norm(w, t)
        v = svd_shrink(w, t * lambda);
    end    

    Problem.cost = @cost;
    function f = cost(w)
        diff = X - w;
        t_norm = reg(L*w);

        f = 1/2 * norm(diff, 'fro') + lambda * t_norm;
    end

    % calculate trace norm
    Problem.reg = @reg;
    function r = reg(w)
        [~,S,~] = svd(w,'econ');
        s = diag(S);
        r = sum(s);
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        g = X - w;
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


    % for shooting
    Problem.shooting_grad = @shooting_grad;
    function g = shooting_grad(w, j, indices)
        g = A(:,j)'* (A(:,indices)* w(indices) - b);
    end


end

