function [] = test_l1_linear_svm()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        %algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'APG-BKT', 'APG-TFOCS-BKT', 'Newton-CHOLESKY', 'NCG-BKT','L-BFGS-TFOCS'};
        algorithms = {'APG-BKT', 'APG-TFOCS-BKT', 'FISTA'};
    end    
    
    
    %% prepare dataset
    if 1     
        % generate synthetic data
        n = 100;    % # of samples per class           
        d = 3;      % # of dimensions
        std = 0.15; % standard deviation 
        l = 2;      % # of classes (must not change)
        
        data = multiclass_data_generator(n, d, l, std);
        d = d + 1; % adding '1' row for intersect
        
        % train data        
        x_train = [data.x_train; ones(1,l*n)];
        % assign y (label) {1,-1}
        y_train(data.y_train<=1.5) = -1;
        y_train(data.y_train>1.5) = 1;

        % test data
        x_test = [data.x_test; ones(1,l*n)];
        % assign y (label) {1,-1}        
        y_test(data.y_test<=1.5) = -1;
        y_test(data.y_test>1.5) = 1;
       
    else
        % load real-world data
        data = importdata('../data/mushroom/mushroom.mat');
        n = size(data.X,1);
        d = size(data.X,2) + 1;         
        x_in = [data.X ones(n,1)]';
        y_in = data.y';
        
        perm_idx = randperm(n);
        x = x_in(:,perm_idx);
        y = y_in(perm_idx);        
        
        % split data into train and test data
        % train data
        n_train = floor(n/8);
        x_train = x(:,1:n_train);
        y_train = y(1:n_train);  
        x_train_class1 = x_train(:,y_train>0);
        x_train_class2 = x_train(:,y_train<0);  
        n_class1 = size(x_train_class1,2);
        n_class2 = size(x_train_class2,2);        
        
        % test data
        x_test = x(:,n_train+1:end);
        y_test = y(n_train+1:end);  
        x_test_class1 = x_test(:,y_test>0);
        x_test_class2 = x_test(:,y_test<0);  
        n_test_class1 = size(x_test_class1,2);
        n_test_class2 = size(x_test_class2,2);    
        n_test = n_test_class1 + n_test_class2;

    end
    lambda = 0.1;
    w_opt = zeros(d,1); 

    
    %% define problem definitions
    problem = l1_linear_svm(x_train, y_train, x_test, y_test, lambda);

    
    %% calculate solution
    if norm(w_opt)
    else
        % calculate solution
        w_opt = problem.calc_solution(problem, 1000);
    end
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);   
    
    
    %% initialize
    w_init = rand(d,1); 
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




