% Setup basic parameters for a quick test
par.beta = 0.96;
par.sigma = 2;
par.alpha = 0.33;
par.delta = 0.10;
cpar.N = 5;
cpar.N_het = 25;
cpar.Nkap = 200;
cpar.min = 0;
cpar.max = 200;
cpar.tol = 1e-6;
cpar.maxit = 500;
par.w = 1;

gri.z = exp([-0.2; -0.1; 0; 0.1; 0.2]);
gri.prob = repmat(1/5, 5, 5); % Dummy transition
gri.k = exp(linspace(log(cpar.min + 1), log(cpar.max + 1), cpar.Nkap))' - 1;

par.r_bar = 0.985 / par.beta - 1;

fprintf('--- TESTING POLICY DIFFERENCE ---\n');

% Case 1: rho = 0
rho1 = 0;
sigma_r2_1 = (1 - rho1^2) * 0.002;
[r_grid1, P_r1] = rouwenhorst(5, 0, rho1, sqrt(sigma_r2_1));
gri_het1.prob = kron(gri.prob, P_r1);
gri_het1.z = kron(gri.z, ones(5, 1));
gri_het1.r = kron(ones(5, 1), r_grid1');
gri_het1.k = gri.k;
sol1 = EGM_het(par, cpar, gri_het1);

% Case 2: rho = 0.9
rho2 = 0.9;
sigma_r2_2 = (1 - rho2^2) * 0.002;
[r_grid2, P_r2] = rouwenhorst(5, 0, rho2, sqrt(sigma_r2_2));
gri_het2.prob = kron(gri.prob, P_r2);
gri_het2.z = kron(gri.z, ones(5, 1));
gri_het2.r = kron(ones(5, 1), r_grid2');
gri_het2.k = gri.k;
sol2 = EGM_het(par, cpar, gri_het2);

fprintf('At k = 200:\n');
fprintf('rho=0.0  | High z, Low r : k'' = %.4f\n', sol1.kpol_k(end, 21));
fprintf('rho=0.0  | High z, High r: k'' = %.4f\n', sol1.kpol_k(end, 25));
fprintf('rho=0.9  | High z, Low r : k'' = %.4f\n', sol2.kpol_k(end, 21));
fprintf('rho=0.9  | High z, High r: k'' = %.4f\n\n', sol2.kpol_k(end, 25));

diff_low = sol2.kpol_k(end, 21) - sol1.kpol_k(end, 21);
diff_high = sol2.kpol_k(end, 25) - sol1.kpol_k(end, 25);
fprintf('Difference (rho=0.9 vs 0.0) -> Low r: %.4f, High r: %.4f\n', diff_low, diff_high);
