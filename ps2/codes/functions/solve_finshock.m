function res = solve_finshock(phi_pi, xi, css_data)
    % Unpack necessary calibrated steady state (CSS) data
    w_css = css_data.w;    r_css = css_data.r;
    A     = css_data.A;    beta  = css_data.beta;
    sigma = css_data.sigma; alpha = css_data.alpha;
    m     = css_data.m;    kappa = css_data.kappa;
    z     = css_data.z;    gamma = css_data.gamma;
    Nb    = css_data.Nb;   bmax  = css_data.bmax_3;  
    Bs    = css_data.Bs;      

    epsilon = 1;              
    pi_ss   = 1 + (0.02/12);
    bmin_new = -2.5;

    r = r_css;
    tol_r = 1e-5;
    err = 10;
    lambda = 0.001;
    max_iter = 1000;
    iter = 0;

    bgrid_new = linspace(bmin_new, bmax, Nb)';

    while abs(err) > tol_r && iter < max_iter
        iter = iter + 1;
        
        % 1. Monetary block
        pi = pi_ss * (r / (r_css * epsilon))^(1 / (phi_pi - 1));
        w  = w_css * (pi_ss/pi)^xi;
        
        % 2. Labor block
        J = (A-w)/(1-beta*(1-sigma));
        theta = (kappa/(beta*m*J))^(-1/alpha);
        f = m*theta^(1-alpha);
        
        % 3. Household block
        ygrid = [w, z];
        Pi = [1-sigma, sigma; f, 1-f];
        
        bp_old = r * [bgrid_new, bgrid_new]; 
        dist_val = 10; iterhh = 0;
        while dist_val > 1e-6 && iterhh < 10000
            iterhh = iterhh + 1;
            [bp_new, ~] = egmUpdateSS(bp_old, bgrid_new, ygrid, Pi, beta, r, gamma, bmin_new);
            dist_val = max(abs(bp_new(:) - bp_old(:)));
            bp_old = bp_new;
        end
        
        % 4. Distribution and Market Clearing
        g_ss_new = solveErgDist(bp_new, bgrid_new, f, sigma);
        Bd = sum(bgrid_new .* g_ss_new, 'all');
        err = Bs - Bd;
        r = r + lambda * err;
    end
    
    % Store results
    res.r = r;
    res.pi = pi;
    res.w = w;
    res.u = sigma / (sigma + f);
    res.iter = iter;
end