function [Problem] = robust_pca(X, mask, lambda)
% This file defines the robust PCA problem. 
%
% Inputs:
%       X           dictionary matrix of size mxn.
%       mask        missing data matrix of size mxn.
%       lambda      l1-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min     ||L||_* + lambda * ||S||_1, 
%           s.t.    L + S = X.
%
% "L" and "S" are the model parameter matrix of size mxn.
%
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on Apr. 27, 2017


    m = size(X, 1);
    n = size(X, 2);
    normX = norm(X, 'fro');
    
    Problem.name = @() 'robust pca';    
    Problem.m = @() m;
    Problem.n = @() n;
    Problem.lambda = @() lambda;
    Problem.X = @() X;  
    Problem.normX = @() normX; 
    Problem.mask = @() mask; 
    
    Problem.prox_trace_norm = @trace_norm_thresh;
    function v = trace_norm_thresh(w, t)
        X(mask) = 0;
        v = svd_shrink(X - w.S + t*w.Y, t);
    end 

    Problem.prox_l1 = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(X - w.L + t*w.Y, t * lambda);
    end 


    % calculate trace norm
    Problem.trace_norm = @trace_norm;
    function r = trace_norm(A)
        [~,S,~] = svd(A,'econ');
        s = diag(S);
        r = sum(s);
    end

    Problem.cardinality = @cardinality;
    function card = cardinality(A)
        card = nnz(A(~mask));
    end

    Problem.cost = @cost;
    function f = cost(w)
        f = trace_norm(w.L) + lambda * norm(w.S,1);
    end

    Problem.residual = @residual;
    function r = residual(w)
        r = X - w.L - w.S;
        r(mask) = 0;
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        error('Not implemted yet.'); 
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
        error('Not implemted yet.'); 
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        error('Not implemted yet.');
    end
end

