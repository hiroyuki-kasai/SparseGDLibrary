function [] = test_elasticnet()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'APG-BKT', 'APG-TFOCS-BKT', 'CD-EasticNet', 'FISTA'}; 
    end    
    
    
    %% prepare dataset
    if 1
        % generate synthtic data        
        n = 500; 
        d = 100; 
        A = randn(d,n); 
        b = randn(d,1); 
        lambda1 = 5;
        lambda2 = 1;
    else
    end
    
    
    %% define problem definitions
    problem = elastic_net(A, b, lambda1, lambda2);

    
    %% initialize
    w_init = rand(n,1); 
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 300;
        options.verbose = true;  

        switch algorithms{alg_idx}
            case {'APG-BKT'}
                
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = apg(problem, options);
                
            case {'APG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = apg(problem, options); 
                
            case {'FISTA'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = fista(problem, options); 
                
            case {'CD-EasticNet'}
                
                options.sub_mode = 'elasticnet';
                [w_list{alg_idx}, info_list{alg_idx}] = cd_lasso_elasticnet(problem, options);                   
                
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
    
    % display iter vs cost/gnorm
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display iter vs. l1 norm, i.e. the toral number of non-zero elements 
    display_graph('iter','l1-norm', algorithms, w_list, info_list); 
    
end




