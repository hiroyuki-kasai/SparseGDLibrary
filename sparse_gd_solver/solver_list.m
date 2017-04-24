function [ algs ] = solver_list(category)
% Return list of solvers.
%
% Inputs:
%       category    category to be returned. 
% Output:
%       algs        list of solvers in the category
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on Apr. 25, 2017


    % supported algorithms by SparseGDLibrary
    fista_algs = {'FISTA'};
    apg_algs = {'APG-BKT','APG-WOLFE','APG-TFOCS','APG-BB-BKT','APG-BB-WOLFE','APG-BB-TFOCS'};
    cd_algs = {'CD-LASSO', 'CD-EasticNet'};
    admm_algs = {'ADMM-LASSO'};
    lasso_algs = {'FISTA','APG-BKT','APG-WOLFE','APG-TFOCS','APG-BB-BKT','APG-BB-WOLFE','APG-BB-TFOCS','CD-LASSO','ADMM-LASSO'};
    elastic_net_algs = {'FISTA','APG-BKT','APG-WOLFE','APG-TFOCS','APG-BB-BKT','APG-BB-WOLFE','APG-BB-TFOCS','CD-EasticNet'};

    
    switch category
        case 'FISTA'
            algs = fista_algs;
        case 'APG'
            algs = apg_algs;  
        case 'CD'
            algs = cd_algs;              
        case 'ADMM'
            algs = admm_algs;   
        case 'LASSO'
            algs = lasso_algs;
        case 'EasticNet'
            algs = elastic_net_algs;          
        case 'ALL'
            algs = [fista_algs, apg_algs, cd_algs, admm_algs];
        otherwise
    end
    
end
