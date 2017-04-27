function [] = test_group_lasso_cv()

    clc;
    clear;
    close all;
    
    
    %% prepare dataset
    if 0
        data = importdata('diabetes.txt');
        b = data.data(:,11);
        A = data.data(:,1:10);
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
        
        % Make mean 0 to eliminate intercept
        A = A - repmat(mean(A,1), [size(A,1) 1]) ;
        b = b - repmat(mean(b,1), [size(b,1) 1]) ;

        % Orthonormalization
        for i=1:length(group_info)
            A(:,group_info{i}) = GSOrth(A(:,group_info{i})) ;
        end
    
    end

    
    %% define parameters for cross-validation
    lamnda_array = [0 10 30 50 100, 200, 300];
    len = length(lamnda_array);

    
    %% prepare arrays for solutions
    W = zeros(d, len);
    l1_norm = zeros(len,1);    
    aprox_err = zeros(len,1);  
    
    
    %% perform cross-validations
    % set options
    clear options;
    options.max_iter = 100;
    options.verbose = true;  
    
    for i=1:len
        lambda = lamnda_array(i);
        % initialize
        if d > n
            options.w_init = zeros(n,1); 
        else
            %w_init = A \ b;
            options.w_init = (A'*A + lambda*eye(d))\(A'*b);
        end
    
        % define problem definitions
        problem = group_lasso_problem(A, b, G, lambda);
        
        [W(:,i), infos] = shooting_grouplasso(problem, options);  
        l1_norm(i) = infos.reg(end);
        aprox_err(i) = infos.cost(end);
    end
    

    %% plot all
    % l1-norm vs coefficient
    display_graph('l1-norm','coeffs', {'Shoooting Group Lasso'}, l1_norm, {W}, 'linear');
    % lambda vs coefficient
    display_graph('lambda','coeffs', {'Shoooting Group Lasso'}, lamnda_array, {W}, 'linear');
    % l1-norm vs approximation error
    display_graph('l1-norm','aprox_err', {'Shoooting Group Lasso'}, l1_norm, {aprox_err}, 'linear');    
    
end




