function solving_EGM = EGM(par, cpar, gri)


% Store the grids we had before

gri.k = gri.k;
gri.z = gri.z;

% Default number of x-grid points if not provided
if ~isfield(cpar, 'Nx')
    cpar.Nx = 100;
end

% Define grid for x

x_min = min(gri.z) * par.w  + par.R * gri.k(1);

x_max = max(gri.z) * par.w + par.R * gri.k(end);

gri.x = linspace(x_min, x_max, cpar.Nx);

kpol_old = zeros(length(gri.x), cpar.N);


% Start loop

for j = 1:cpar.N
    
    kpol_old(:, j) = 0.5 * gri.x';

end

diff = cpar.tol + 1;
cpar.maxit = 1000;
iter = 0;

while diff > cpar.tol && iter < cpar.maxit
    % Update policy function and calculate the difference
    kpol_new = zeros(size(kpol_old));

    for j = 1:cpar.N

        c_tod = zeros(cpar.Nkap, 1);
        x_end = zeros(cpar.Nkap, 1);

        for kp = 1:cpar.Nkap

            exp_MU = 0;

            for z = 1:cpar.N

                x_tom = gri.z(z) * par.w + par.R * gri.k(kp);

                k_tom_tom = interp1(gri.x, kpol_old(:,z), x_tom, 'linear', 'extrap');

                k_tom_tom = max(0, min(k_tom_tom, x_tom - 1e-10));

                c_tom = x_tom - k_tom_tom;

                prob_transition = gri.prob(j, z);
                exp_MU = exp_MU + prob_transition * (c_tom^(-par.sigma));

            end

            c_tod(kp) = (par.beta * par.R *exp_MU)^(-1/par.sigma);
            x_end(kp) = gri.k(kp) + c_tod(kp);

        end

        % Borrowing constraint
        
        x_end_bc = [0; x_end];

        if isrow(gri.k)
            k_with_BC = [0; gri.k']; 
        else
            k_with_BC = [0; gri.k];
        end
        
        kpol_new(:, j) = interp1(x_end_bc, k_with_BC, gri.x, "linear","extrap");

        kpol_new(:, j) = max(0, min(kpol_new(:, j), gri.x'));
    end
    
    diff = max(abs(kpol_new - kpol_old), [], 'all');
    kpol_old = kpol_new;
    iter = iter + 1;

    if mod(iter, 50) == 0
        fprintf('EGM Iteration: %d, Diff: %e\n', iter, diff);
    end

end

fprintf('EGM converged successfully in %d iterations!\n', iter);

% Store the final policy in the output struct
solving_EGM.kpol_x = kpol_new;
solving_EGM.x = gri.x;  % export x-grid so the caller can use it

% We can also easily recover the consumption policy on the x grid
% Because x = c + k'  =>  c = x - k'
solving_EGM.cpol_x = repmat(gri.x', 1, cpar.N) - kpol_new;

% To make this comparable with your VFI from Question 1.A, you might want to 
% map the policy back to the (k, z) state space instead of (x, z). 
solving_EGM.kpol_k = zeros(cpar.Nkap, cpar.N);
solving_EGM.cpol_k = zeros(cpar.Nkap, cpar.N);

for j = 1:cpar.N
    % Cash-on-hand for each k grid point today
    x_today = gri.z(j) * par.w + par.R * gri.k; 
    x_today = x_today(:);  % force column to match kpol_k(:,j)
    
    % Interpolate the x-policies onto the k-grid
    solving_EGM.kpol_k(:, j) = interp1(gri.x, solving_EGM.kpol_x(:, j), x_today, 'linear', 'extrap');
    solving_EGM.cpol_k(:, j) = interp1(gri.x, solving_EGM.cpol_x(:, j), x_today, 'linear', 'extrap');
end

end