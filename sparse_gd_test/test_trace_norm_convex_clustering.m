function [] = test_trace_norm_convex_clustering()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    weighted = 1;
    lambda_weight = 300;
    %dataset = 'synthetic';
    dataset = 'synthetic_outlier';
    %dataset = 'real';

    
    %% prepare dataset
    if strcmp(dataset, 'synthetic')
        fprintf('generating syntheric dataset ....');
        class_num = 4;
        n_per_class = 30;
        n = class_num * n_per_class;
        d = 5; 
        interval = 5;

        X = [];
        label = zeros(1, n);
        cnt = 0;
        for i=1:class_num
            X = vertcat(X, 1.0*randn(n_per_class,d) + interval*(i-1)); 
            for k=1:n_per_class
                cnt = cnt + 1;
                label(1, cnt) = i;
            end
        end
        fprintf('finished.\n');
    elseif strcmp(dataset, 'synthetic_outlier')
        class_num = 4;
        n_per_class = 5;
        n = class_num * n_per_class;
        d = 2; 
        d_outlier = 2;
        rnggn = 2;
        
        [X,~] = syndata_c(d, class_num, n, rnggn, d_outlier);
        label = zeros(1, n);
        cnt = 0;        
        for i=1:class_num
            for k=1:n_per_class
                cnt = cnt + 1;
                label(1, cnt) = i;
            end
        end
            
        X = X';
        
    else
        fprintf('loading dataset ....');
        data = importdata('../data/clustering/iris.mat');
        X = data.X';
        label = data.y; 
        n = size(X, 1);
        d = size(X, 2);
        class_num = length(unique(data.y));
        fprintf('finished.\n');
    end
    
    % generate difference matrix
    L = generate_difference_matrix(n, d);
    
    % generate similarity matrix
    if weighted
        W_for_L = generate_similarity_matrix(X', n, 'nn');
    else
        W_for_L = ones(n*(n-1)/2, 1);
    end    
    
    % set lammbda
    lambda = norm(std(X)) * lambda_weight; 
    
    %[P_output, fv_primal, fv_dual] = cvxclus_dual(X', 1, 0, 200);
    

    %% define problem definitions
    problem = trace_norm_convex_clustering(X, L, W_for_L, lambda);

    
    %% admm_trace_norm
    clear options;
    options.w_init = X;
    options.tol_gnorm = -Inf;
    options.max_iter = 2000;
    options.verbose = true; 
    options.store_w = true;
    options.class_num = class_num;
    options.mode = 'clustering';
    options.round_precision = 0;
    options.stopfun = @clustering_stopfun;
    [w_admmtn, info_admmtn] = admm_trace_norm(problem, options);
                
    
    %% K-means
    [k_means_label, k_means_center] = litekmeans(X, class_num, 'MaxIter', 100);  
    
    
    
    
    
    %% plot all
    close all;
    
    % display iter vs. cost
    display_graph('iter','cost', {'ADMM-TRACE-NORM'}, {w_admmtn}, {info_admmtn});
    % display iter vs. trace (nuclear) norm
    display_graph('iter','trace_norm', {'ADMM-TRACE-NORM'}, {w_admmtn}, {info_admmtn}); 
    
    
    %% plot
    figure;
    color_style = {'b','g','m','y','k'};
    fs = 16;
    h_cell = cell(2*class_num, 1);
    legend_str = cell(2*class_num+2, 1);
    legend_h_cell = [];    
    
    
    % convex clustering
    w_history = info_admmtn.w;
    w_res = reshape(w_history(:,1), [n d]);
    % draw initial points
    plot_type = 'o';
    line_width = 1;
    mark_size = 8;    
    for plot_idx=1:n
        class_id = label(plot_idx);
        h_cell{class_id} = plot(w_res(plot_idx,1), w_res(plot_idx,2), 'MarkerSize', mark_size, 'Marker', plot_type, 'Color', color_style{class_id}, 'LineWidth', line_width); 
        hold on;
        drawnow;
    end
    
    for k=1:class_num
        legend_str{k}   = sprintf('class %d (Initial)', k);
        legend_h_cell   = [legend_h_cell h_cell{k}];
    end     

    % draw trace points    
    plot_type = '+';
    len = size(w_history, 2);
    period = floor(len/10);
    mark_size = 2;
    line_width = 1;
    for iter_idx=2:period:len
        w_res = reshape(w_history(:,iter_idx), [n d]);
        for plot_idx=1:n
            class_id = label(plot_idx);
            h_cell{class_id+class_num} = plot(w_res(plot_idx,1), w_res(plot_idx,2), 'MarkerSize', mark_size, 'Marker', plot_type, 'Color', color_style{class_id}, 'LineWidth', line_width); 
            hold on;
            drawnow;
        end
        
        fprintf('%d\n', iter_idx);
        drawnow;  
    end


    
    for k=class_num+1:2*class_num
        legend_str{k}   = sprintf('class %d (Sequence)', k-class_num);
        legend_h_cell   = [legend_h_cell h_cell{k}];
    end     
    
    % draw final points  
    plot_type =  'o';
    mark_color = 'red';
    mark_size = 10;
    line_width = 2;
    for plot_idx=1:n
        h_cell{2*class_num+1} = plot(w_res(plot_idx,1), w_res(plot_idx,2), 'MarkerSize', mark_size, 'Marker', plot_type, 'Color', mark_color, 'LineWidth', line_width); 
        hold on;
    end
    legend_str{2*class_num+1} = 'convex clustering';
    legend_h_cell   = [legend_h_cell h_cell{2*class_num+1}];    
    
    % k-means centroid
    plot_type =  'x';
    mark_color = 'cyan';
    mark_size = 10;
    line_width = 2;
    for cls_idx=1:class_num
        h_cell{2*class_num+2} = plot(k_means_center(cls_idx,1), k_means_center(cls_idx,2), 'MarkerSize', mark_size, 'Marker', plot_type, 'Color', mark_color, 'LineWidth', line_width); 
        hold on;
    end
    hold off;
    legend_str{2*class_num+2} = 'k-means';
    legend_h_cell   = [legend_h_cell h_cell{2*class_num+2}];
    
    
    % draw legend and labels
    ax1 = gca;
    set(ax1,'FontSize',fs);    
    legend(legend_h_cell, legend_str, 'Location', 'best');
    xlabel(ax1,'x1','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'x2','FontName','Arial','FontSize',fs,'FontWeight','bold');
    
    
end


function L = generate_difference_matrix(n, d)

    %L = zeros(n, d);
    L = zeros(n*(n-1)/2, n);

    point_idx = 1;
    for row=1:n
        for col=row+1:n
            L(point_idx, row) = 1;
            L(point_idx, col) = -1;
            point_idx = point_idx+1;
        end
    end

end


function W_for_L = generate_similarity_matrix(X, n, type)

    dim = n*(n-1)/2;

    % generate similarity graph by codes written by Ingo Buerk.
    switch type
        case 'full'
            W = SimGraph_Full(X, 1);
        case 'nn'
            k = 5;
            Type = 1; % 1: Normal, 2: Mutual
            W = SimGraph_NearestNeighbors(X, k, Type, 1);
        case 'epsilon'
            epsilon = 0.001;
            SimGraph_Epsilon(M, epsilon);
        otherwise
            W = SimGraph_Full(X, 1);
    end

    W_for_L = zeros(dim, 1);
    cnt = 0;
    for row = 1:n
        for col = row+1:n
            cnt = cnt + 1;
            W_for_L(cnt, 1) = W(row, col);
        end
    end

end


function stop = clustering_stopfun(problem, w, ~, stop_options)
    dim = problem.dim();
    stop = false;

    if strcmp(stop_options.mode, 'clustering')
        rounded_w = round(w, stop_options.round_precision);
        unique_rounded_w = unique(rounded_w, 'rows');
        num_unique_rows = size(unique_rounded_w, 1);

        if stop_options.verbose
            fprintf('current class number: %d\n', num_unique_rows);        
        end

        if num_unique_rows == stop_options.class_num
            if stop_options.verbose
                fprintf('Reached target class number %d under the decimal point of significant digits %d.\n', stop_options.class_num, stop_options.round_precision);

                fprintf('Final cluster centroids:\n');
                for c_idx=1:stop_options.class_num
                    fprintf('\t(');
                    for d_idx=1:dim
                        fprintf('%5.2f ', unique_rounded_w(c_idx, d_idx));
                    end
                    fprintf(')\n');
                end
                fprintf('\n');
            end
            stop = true;
        end  
    end
end
