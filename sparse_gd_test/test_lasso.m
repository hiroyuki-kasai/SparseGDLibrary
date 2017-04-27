function [] = test_lasso()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'APG-TFOCS-BKT', 'CD-LASSO', 'FISTA', 'ADMM-LASSO'}; 
        %algorithms = {'APG-TFOCS-BKT', 'FISTA', 'ADMM-LASSO', 'CD-LASSO'}; 
    end    
    
    
    %% prepare dataset
    n = 1280; 
    d = 100;  
    k = 15; 
    noise_level = 0.01;
    [A, b, x0, lambda] = generate_lasso_data(n, d, k, noise_level);    
    
    
    %% define problem definitions
    problem = lasso_problem(A, b, lambda);

    
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
        options.solution = x0;        
        options.tol_gnorm = 1e-10;
        options.max_iter = 100;
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
                
            case {'ADMM-LASSO'}
                
                options.rho = 0.1;
                [w_list{alg_idx}, info_list{alg_idx}] = admm_lasso(problem, options);    
                
            case {'CD-LASSO'}
                
                options.sub_mode = 'lasso';
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
    % display iter vs. cost
    display_graph('iter','sol_optimality_gap', algorithms, w_list, info_list);        
    % display iter vs. l1-norm
    display_graph('iter','l1-norm', algorithms, w_list, info_list, 'linear');
    % display coeff position vs. coeff amplitude
    algorithms{alg_idx+1} = 'Original';
    w_list{alg_idx+1} = x0;
    display_graph('coeff_pos','coeff_amp', algorithms, w_list, w_list, 'linear', 'line-with-mark', 1);  
    
    
end




