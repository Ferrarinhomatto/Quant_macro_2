function [V, cpol, kpol] = VFI_GS(par, gri, cpar)

% Start timer
tic;

% initiate function matrices
V = zeros(cpar.N, cpar.Nkap);
V_new = zeros(cpar.N, cpar.Nkap);
cpol = zeros(cpar.N, cpar.Nkap);
kpol = zeros(cpar.N, cpar.Nkap);

% Store the grids
gri.k = gri.k;
gri.z = gri.z;

% Parameters for the loop
iter = 0;
diff = cpar.tol + 1;

cpar.maxit = 1500;

fprintf('Starting VFI with Optimized Golden Section Search...\n');

% Initiate Main Loop
while iter < cpar.maxit && diff > cpar.tol
   
    iter = iter + 1;
   
    for j = 1:cpar.N

        z_current = gri.z(j);

       for i = 1:cpar.Nkap
        
        k_current = gri.k(i);
        
        % Objective function
        obj_func = @(C) bellman_RHS(C, k_current, z_current, par, ...
            cpar, gri, j, V);

        % Bounds for C
        c_min = 1e-6;
       
        c_max = par.R * k_current + z_current * par.w;

        % Use golden search
        [c_opt, V_new(j,i)] = MaxGoldenSearch(obj_func, c_min, c_max, ...
            cpar.tol, cpar.maxit);

        % debugging
        if iter == 1 && j == 1 && i <= 3
             fprintf('Iter1 state (j=%d,i=%d): c_opt=%.4f V_new=%.4f\n', ...
                 j, i, c_opt, V_new(j,i));
        end

        cpol(j,i) = c_opt;
        kpol(j,i) = k_motion(k_current, z_current, cpol(j,i), par);

        % store the value of the new value function
        V_new(j,i) = V_new(j,i);

       end

    end  % close the FOR loops

    % Convergence check
    diff = max(abs(V_new(:)- V(:)));
    V = V_new;

    % For clarity in the iterations
    if iter <= 5 || mod(iter, 10) == 0
        fprintf('Iter %d: diff = %.8e, V_min = %.4f, V_max = %.4f\n', ...
                iter, diff, min(V_new(:)), max(V_new(:)));
    end

end

% Calculate elapsed time
vfi_time = toc;

% Convergence check and reporting
if diff <= cpar.tol
    fprintf('VFI converged after %d iterations (sup norm: %.8f)\n', ...
        iter, diff);
else
    fprintf('VFI did not converge after %d iterations (sup norm: %.8f)\n',...
        cpar.maxit, diff);
end

% Display timing information
fprintf('VFI execution time: %.2f seconds\n', vfi_time);

% Display iterations per second
if vfi_time > 0
    fprintf('Average iteration speed: %.2f iterations/second\n', iter/vfi_time);
end

end