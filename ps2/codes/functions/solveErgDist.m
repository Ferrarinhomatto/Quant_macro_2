function g_ss = solveErgDist(bp, bgrid, f, sigma)
%SOLVEERGDIST  Stationary distribution via iterating gUpdate to convergence.
%
%   g_ss = solveErgDist(bp, bgrid, f, sigma)
%
% Inputs
%   bp    : (Nb x 2) steady-state policy b'(b,x)
%   bgrid : (Nb x 1) asset grid
%   f     : job-finding probability (U->E)
%   sigma : separation probability (E->U)
%
% Output
%   g_ss  : (Nb x 2) stationary distribution over (b,x)
%
% Method:
%   - initialise g0 as uniform over b with the invariant x-marginal
%   - iterate g_{n+1} = gUpdate(g_n, ...) until sup norm converges


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0) Input checks / coercions
% NOTE: This section has some very strict checks to make sure you
% don't input incorrect things into the function. 

if nargin ~= 4
    error('solveErgDist:BadNargin', 'Expected 4 inputs: bp, bgrid, f, sigma.');
end

if ~isnumeric(bgrid) || ~isvector(bgrid) || isempty(bgrid)
    error('solveErgDist:BadBgrid', 'bgrid must be a nonempty numeric vector.');
end
bgrid = double(bgrid(:));
Nb = numel(bgrid);
if any(diff(bgrid) <= 0)
    error('solveErgDist:NonMonotoneGrid', 'bgrid must be strictly increasing.');
end

if ~isnumeric(bp) || ~ismatrix(bp) || any(~isfinite(bp(:)))
    error('solveErgDist:BadBP', 'bp must be a finite numeric 2D array.');
end
bp = double(bp);
if isequal(size(bp), [2, Nb])
    bp = bp.';
end
if ~isequal(size(bp), [Nb, 2])
    error('solveErgDist:BadBPSize', 'bp must be size Nb-by-2 = %d-by-2.', Nb);
end

if ~isnumeric(f) || ~isscalar(f) || ~isfinite(f) || f < 0 || f > 1
    error('solveErgDist:BadF', 'f must be a finite scalar in [0,1].');
end
if ~isnumeric(sigma) || ~isscalar(sigma) || ~isfinite(sigma) || sigma < 0 || sigma > 1
    error('solveErgDist:BadSigma', 'sigma must be a finite scalar in [0,1].');
end
f = double(f); sigma = double(sigma);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Controls for iteration
tol   = 1e-10;
maxit = 200000;
verbose = false; % <- FOR NO NOISE


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Initial guess g0

% Invariant x-marginal for the 2-state chain:
% u = sigma/(sigma+f), e = f/(sigma+f)
if (sigma + f) <= 0
    error('solveErgDist:NoTransitions', 'sigma+f must be > 0 to have a well-defined invariant distribution.');
end
u = sigma / (sigma + f);
e = 1 - u;

% Start uniform over b within each x, with correct x-marginals
g = ones(Nb, 2);
g(:,1) = g(:,1) * (e / sum(g(:,1)));
g(:,2) = g(:,2) * (u / sum(g(:,2)));


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) Iterate to convergence

dist = inf;
it = 0;

if verbose
    fprintf('Iterating stationary distribution...\n');
    fprintf('%6s  %14s\n', 'iter', 'max|gNext-g|');
end

while (dist > tol) && (it < maxit)
    it = it + 1;

    gNext = gUpdate(g, bp, bgrid, f, sigma);

    dist = max(abs(gNext(:) - g(:)));
    g = gNext;

    if verbose && (mod(it,25)==0 || it==1)
        fprintf('%6d  %14.6e\n', it, dist);
    end
end

if dist > tol
    warning('solveErgDist:NoConvergence', ...
        'Distribution iteration hit maxit=%d with dist=%g.', maxit, dist);
elseif verbose
    fprintf('Distribution converged in %d iterations (dist=%g).\n', it, dist);
end

% Output
g_ss = g;

% Small sanity check: mass ~ 1
mass = sum(g_ss(:));
if abs(mass - 1) > 1e-10
    % not fatal; depends on whether you interpret g as pdf or raw mass
    warning('solveErgDist:MassNotOne', 'sum(g_ss(:))=%g (expected 1).', mass);
end

