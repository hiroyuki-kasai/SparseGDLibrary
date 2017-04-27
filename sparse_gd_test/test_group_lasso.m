function [] = test_group_lasso()

    clc;
    clear;
    close all;
    
    %rng('default');
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'SHOOTING-GROUPLASSO'};
    end    
    
    
    %% prepare dataset
    if 0
        data = importdata('diabetes.txt');
        b = data.data(:,11);
        A = data.data(:,1:10);
        lambda = 2500;
        [n,d] = size(A);
        G = [1 1 2 2 3 3 3 3 3 3];
    else
        % load data
        data = importdata('test_data_for_glasso.mat');
        A = data.X;
        b = data.Y;
        group_info = data.group;
        count = 0;
        for i=1:length(group_info)
            for j=1:length(group_info{i})
                count = count  + 1;
                G(count) = i;
            end
        end
        [n,d] = size(A);
        % lasso parameter
        lambda = 10;
        
        % Make mean 0 to eliminate intercept
        A = A - repmat(mean(A,1), [size(A,1) 1]) ;
        b = b - repmat(mean(b,1), [size(b,1) 1]) ;

        % Orthonormalization
        for i=1:length(group_info)
            A(:,group_info{i}) = GSOrth(A(:,group_info{i})) ;
        end
    
    end

    
    %% define problem definitions
    problem = group_lasso_problem(A, b, G, lambda);

    
    %% initialize
    if d > n
        w_init = zeros(n,1); 
    else
        %w_init = A \ b;
        w_init = (A'*A + lambda*eye(d))\(A'*b);
    end

    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        %options.solution = x0;        
        options.tol_gnorm = 1e-24;
        options.max_iter = 100;
        options.verbose = true;  

        switch algorithms{alg_idx}
            case {'SHOOTING-GROUPLASSO'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = shooting_grouplasso(problem, options);                
                
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
    % display iter vs. group lasso norm
    display_graph('iter','reg', algorithms, w_list, info_list, 'linear');   
    
end




