function [x_opt, f_opt] = MaxGoldenSearch(func, c_low, c_high, tol, max_iter)
% GOLDEN_SECTION_OPTIMIZED - High-performance Golden Section Search
%
% This version is optimized for speed with:
% - Minimal function evaluations
% - Early termination conditions
% - Proper convergence logic
% - Function value caching
%
% INPUTS:
%   func      - function handle to maximize
%   c_low     - lower bound of search interval
%   c_high    - upper bound of search interval
%   tol       - convergence tolerance
%   max_iter  - maximum iterations
%
% OUTPUTS:
%   x_opt     - optimal point
%   f_opt     - function value at optimal point

    % Pre-compute golden ratio constant
    gr = (sqrt(5) - 1) / 2; 
    
    % Initialize interior points
    c = c_high - gr * (c_high - c_low);
    d = c_low + gr * (c_high - c_low);
    
    % Initial function evaluations
    fc = func(c);
    fd = func(d);
    
    % Iteration counter
    iter = 0;
    
    % Main iteration loop - FIXED: use AND operator
    while abs(c_high - c_low) > tol && iter < max_iter
        iter = iter + 1;
        
        if fc > fd
            % Maximum is in [a_low, d]
            c_high = d;
            d = c;
            fd = fc;  % Reuse function evaluation - NO new func call needed!
            c = c_high - gr * (c_high - c_low);
            fc = func(c);  % Only ONE new function evaluation per iteration
        else
            % Maximum is in [c, a_high]
            c_low = c;
            c = d;
            fc = fd;  % Reuse function evaluation - NO new func call needed!
            d = c_low + gr * (c_high - c_low);
            fd = func(d);  % Only ONE new function evaluation per iteration
        end
        
    end
    
    % Return the better point
    if fc > fd
        x_opt = c;
        f_opt = fc;
    else
        x_opt = d;
        f_opt = fd;
    end

end