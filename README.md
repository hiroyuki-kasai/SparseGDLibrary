# SparseGDLibrary : Sparse Gradient Descent Library in MATLAB
----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/)

Last page update: April 24, 2017

Latest library version: 1.0.0 (see Release notes for more info)

Introduction
----------
The SparseGDLibrary is a **pure-Matlab** library of a collection of **unconstrained optimization algorithms** for **sparse modeling**. 


List of sparse gradient algorithms available in SparseGDLibrary
---------
- **APG** (Accelerated gradient descent, i.e., Nesterov AGD)
- **[FISTA](http://epubs.siam.org/doi/abs/10.1137/080716542)** (Fast iterative shrinkage-thresholding algorithm)
- **[CD](https://en.wikipedia.org/wiki/Coordinate_descent)** (Coodinate descent) **for Lasso and Elastic Net** 
- **[ADMM](http://stanford.edu/~boyd/admm.html)** (The alternating direction method of multipliers) **for Lasso**

List of [line-search](https://en.wikipedia.org/wiki/Line_search) algorithms available in SparseGDLibrary
---------
- **[Backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search)** (a.k.a Armijo condition)
- **[Strong wolfe line search](https://en.wikipedia.org/wiki/Wolfe_conditions)**
- **Exact line search**
    - Only for quadratic problem.
- **[TFOCS](http://cvxr.com/tfocs/)-style line search**

Supported problems
---------
* [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics) (Least absolute shrinkage and selection operator) problem
* [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization) problem
* [Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) problem with trace norm minimization 
* L1-norm logistic regression
* L1-norm linear [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM)

Folders and files
---------

<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script to check and understand this package easily. 
./demo_lasso_cv.m       - Demonstration script for lasso problem with cross validation. 
|plotter/               - Contains plotting tools to show convergence results and various plots.
|tool/                  - Some auxiliary tools for this project.
|problem/               - Problem definition files to be solved.
|gd_solver/             - Contains various gradient descent optimization algorithms.
|gd_test/               - Some helpful test scripts to use this package.
</pre>
                                 

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

Usage example 1 ([Lasso problem](https://en.wikipedia.org/wiki/Lasso_(statistics)))
----------------------------
Now, just execute `demo` for demonstration of this package.
```Matlab
%% Execute the demonstration script
demo; 
```

The "**demo.m**" file contains below.
```Matlab
% set number of dimensions, cardinality and noise level.
n = 1280; 
d = 100;  
k = 15; 
noise_level = 0.01;

% generate dataset
[A, b, x0, lambda] = generate_lasso_data(n, d, k, noise_level);

% define problem
problem = lasso(A, b, lambda);

% perform algorithms (FISTA and ADMM)
options.w_init = zeros(n,1); 
[w_fista, info_fista] = fista(problem, options);
[w_admm, info_admm] = admm_lasso(problem, options); 

% plot all
% display iter vs. cost
display_graph('iter','cost', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm});
% display iter vs. l1-norm
display_graph('iter','sol_optimality_gap', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm});  
```

* Output results 

<img src="https://dl.dropboxusercontent.com/u/869853/Github/SparseGDLibrary/images/iter_cost_solgap.png" width="900">
<br /><br />



Usage example 1: more plots
----------------------------

The decrease of the l1-norm of the solution according to iterations is illustrated.

```Matlab
display_graph('iter','l1-norm', {'FISTA', 'ADMM'}, {w_fista, w_admm}, {info_fista, info_admm}, 'linear');
```

The final coefficients in each position of the solution are displayed in comparisn with the ogirinal input sparse signal.

```Matlab
display_graph('coeff_pos','coeff_amp', {'FISTA','ADMM','Original (x0)'}, {w_fista, w_admm, x0}, {w_fista, w_admm, x0}, 'linear', 'line-with-mark', 1);           
```

* Output results 

<img src="https://dl.dropboxusercontent.com/u/869853/Github/SparseGDLibrary/images/coeff_pos_amp.png" width="500">
<br /><br />

Usage example 2 ([Lasso problem](https://en.wikipedia.org/wiki/Lasso_(statistics)) with cross-validation)
----------------------------
Execute `demo_lasso_cv` for the lasso problem with cross-validation.

```Matlab
% Execute the demonstration script
demo_lass_cv; 
```

The "**demo_lass_cv.m**" file contains below.
```Matlab
function demo_lasso_cv()

% set number of dimensions, cardinality and noise level.
n = 1280; 
d = 100;  
k = 15; 
noise_level = 0.01;

% generate dataset
[A, b, ~, ~, lambda_max] = generate_lasso_data(n, d, k, noise_level);

% set algorithms and solver (e.g., FISTA)
algorithm = {'FISTA'};

% initialize
% define parameters for cross-validation
num_cv = 10;
lambda_unit = lambda_max/num_cv;
lambda_array = 0+lambda_unit:lambda_unit:lambda_max;

% prepare arrays for solutions
W = zeros(n, num_cv);
l1_norm = zeros(num_cv,1);    
aprox_err = zeros(num_cv,1);  

% set options
options.w_init = zeros(n,1); 

% perform cross-validations
for i=1:length(lambda_array)
    lambda = lambda_array(i);
    problem = lasso(A, b, lambda);

    [W(:,i), infos] = fista(problem, options);
    l1_norm(i) = infos.reg(end);
    aprox_err(i) = infos.cost(end);
end

% plot all
% display l1-norm vs. coefficient
display_graph('l1-norm','coeffs', algorithm, l1_norm, {W}, 'linear');
% display lambda vs. coefficient
display_graph('lambda','coeffs', algorithm, lambda_array, {W}, 'linear');
% display l1-norm vs. approximation error
display_graph('l1-norm','aprox_err', algorithm, l1_norm, {aprox_err}, 'linear');

end  

```

* Output results 

<img src="https://dl.dropboxusercontent.com/u/869853/github/SparseGDLibrary/images/lasso_cv.png" width="900">
<br /><br />


License
-------
The SparseGDLibrary is free and open source for academic/research purposes (non-commercial).


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

Release Notes
--------------

* Version 1.0.0 (Nov. 24, 2017)
    - Initial version.

