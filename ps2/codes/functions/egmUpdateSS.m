function [bp_new, c] = egmUpdateSS(bp_old, bgrid, ygrid, Pi, beta, r, gamma, bmin)
%EGMUPDATESS  One steady-state EGM update for b'(b,x) in the 2-state model.
%
%   [bp_new, c] = egmUpdateSS(bp_old, bgrid, ygrid, Pi, beta, r, gamma, bmin)
%
% Inputs
%   bp_old : (Nb x 2) current guess for saving policy b'(b,x)
%   bgrid  : (Nb x 1) asset grid (strictly increasing); also used as grid for b'
%   ygrid  : (1 x Nx) income by state: [y(e), y(u)] = [w, z]
%   Pi     : (Nx x Nx) transition matrix; Pi(j,jp) = P(x'=jp | x=j)
%   beta   : scalar discount factor, in (0,1)
%   r      : scalar gross real interest rate (> 0)
%   gamma  : scalar CRRA coefficient (> 0)
%   bmin   : scalar borrowing limit; b' >= bmin is imposed
%
% Outputs
%   bp_new : (Nb x 2) updated saving policy b'(b,x)
%   c      : (Nb x 2) consumption policy c(b,x) implied by bp_new
%
% Method:
%   1) Compute next-period consumption c'(b',x') from bp_old and budget constraint
%   2) Apply Euler equation to recover current consumption c(b',x) on the b' grid
%   3) Back out endogenous current assets tilde_b(b',x) via budget constraint
%   4) Interpolate b'(b,x) back onto the exogenous asset grid bgrid
%   5) Impose borrowing constraint b' >= bmin and upper bound b' <= bgrid(end)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0) Input checks / coercions
% NOTE: This section has some very strict checks to make sure you
% don't input incorrect things into the function. 

if nargin ~= 8
    error('egmUpdateSS:BadNargin', ...
        'Expected 8 inputs: bp_old, bgrid, ygrid, Pi, beta, r, gamma, bmin.');
end

% bgrid
if ~isnumeric(bgrid) || ~isvector(bgrid) || isempty(bgrid)
    error('egmUpdateSS:BadBgrid', 'bgrid must be a nonempty numeric vector.');
end
bgrid = double(bgrid(:));  % force column
Nb = numel(bgrid);
if any(~isfinite(bgrid))
    error('egmUpdateSS:BadBgrid', 'bgrid contains non-finite values.');
end
if any(diff(bgrid) <= 0)
    error('egmUpdateSS:NonMonotoneGrid', 'bgrid must be strictly increasing.');
end
bmax = bgrid(end);

% ygrid
if ~isnumeric(ygrid) || ~isvector(ygrid) || isempty(ygrid)
    error('egmUpdateSS:BadYgrid', 'ygrid must be a nonempty numeric vector.');
end
ygrid = double(ygrid(:))';  % force row vector
Nx = numel(ygrid);

% Pi
if ~isnumeric(Pi) || ~ismatrix(Pi) || any(~isfinite(Pi(:)))
    error('egmUpdateSS:BadPi', 'Pi must be a finite numeric 2D matrix.');
end
Pi = double(Pi);
if ~isequal(size(Pi), [Nx, Nx])
    error('egmUpdateSS:BadPiSize', 'Pi must be size Nx-by-Nx where Nx=numel(ygrid)=%d.', Nx);
end
if any(abs(sum(Pi,2) - 1) > 1e-10)
    error('egmUpdateSS:PiRowSums', 'Rows of Pi must sum to 1.');
end
if any(Pi(:) < -1e-12)
    error('egmUpdateSS:PiNegative', 'Pi contains negative entries.');
end

% Scalars
for name = ["beta","r","gamma","bmin"]
    val = eval(name);
    if ~isnumeric(val) || ~isscalar(val) || ~isfinite(val)
        error('egmUpdateSS:BadScalar', '%s must be a finite numeric scalar.', name);
    end
end
beta  = double(beta);
r     = double(r);
gamma = double(gamma);
bmin  = double(bmin);

if beta <= 0 || beta >= 1
    error('egmUpdateSS:BadBeta', 'beta must be in (0,1).');
end
if r <= 0
    error('egmUpdateSS:BadR', 'r must be > 0 (gross interest rate).');
end
if gamma <= 0
    error('egmUpdateSS:BadGamma', 'gamma must be > 0.');
end

% bp_old
if ~isnumeric(bp_old) || ~ismatrix(bp_old) || any(~isfinite(bp_old(:)))
    error('egmUpdateSS:BadBP', 'bp_old must be a finite numeric 2D array.');
end
bp_old = double(bp_old);

% Accept either Nb-by-Nx or Nx-by-Nb and transpose if needed
if isequal(size(bp_old), [Nx, Nb])
    bp_old = bp_old.'; % -> Nb-by-Nx
end
if ~isequal(size(bp_old), [Nb, Nx])
    error('egmUpdateSS:BadBPSize', ...
        'bp_old must be size Nb-by-Nx = %d-by-%d (or %d-by-%d, which will be transposed).', ...
        Nb, Nx, Nx, Nb);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Next-period consumption on the b' grid: c'(b',x')

% Impose that lower limit is either borrowing constraint or min of bgrid
bmin = max(min(bgrid),bmin);
% Enforce practical bounds on bp_old (helps avoid issues during early iterations)
bp_old = min(max(bp_old, bmin), bmax);

% Here b' = bgrid(i') and bp_old(i',j') is b''.
c_next = ygrid + r * bgrid - bp_old;

% Consumption must be strictly positive for CRRA marginal utility
cmin = min(c_next(:));
if cmin <= 0
    error('egmUpdateSS:NonPositiveCnext', ...
        'Non-positive c_next encountered (min=%g). Check initial bp_old / bounds / parameters.', cmin);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Euler equation => current consumption on the b' grid: c(b',x)

% For each current state j, Emu(:,j) = sum_{jp} Pi(j,jp)*mu_next(:,jp)
mu_next = c_next .^ (-gamma); % u'(c') = c'^(-gamma)
Emu = mu_next * (Pi.');   % (Nb x Nx) * (Nx x Nx) = (Nb x Nx), columns indexed by current j
c_endo = (beta * r * Emu) .^ (-1/gamma);

cmin2 = min(c_endo(:));
if cmin2 <= 0 || any(~isfinite(c_endo(:)))
    error('egmUpdateSS:BadCendo', 'Bad c_endo computed (min=%g).', cmin2);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) Endogenous grid: \tilde b(b',x)

% \tilde b = (c + b' - y)/r  with b' = bgrid(i')
b_endo = (c_endo + bgrid - ygrid) / r;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4) Interpolate back onto original grid (for each j)

bp_new = zeros(Nb, Nx);

for j = 1:Nx
    tb = b_endo(:, j);        % endogenous current assets corresponding to b'=bgrid
    bprime_nodes = bgrid;     % associated b' values

    % EGM theory implies tb is increasing in b'; numerically it can have tiny violations.
    % Make interpolation robust by sorting and removing duplicates.
    [tb_s, idx] = sort(tb);
    bprime_s = bprime_nodes(idx);

    % Remove duplicate tb_s values (interp1 requires strictly monotone x for best behaviour)
    [tb_u, ia] = unique(tb_s, 'stable');
    bprime_u = bprime_s(ia);

    if numel(tb_u) < 2
        error('egmUpdateSS:DegenerateEndogenousGrid', ...
            'Endogenous grid for state j=%d is degenerate (too few unique points).', j);
    end

    % Interpolate b'(b,x=j) onto exogenous bgrid
    % Use linear interpolation; for out-of-range, fill with endpoints (then enforce constraints).
    bp_j = interp1(tb_u, bprime_u, bgrid, 'linear', 'extrap');

    % Borrowing constraint + top-of-grid cap (as in the appendix footnote)
    bp_j = min(max(bp_j, bmin), bmax);

    bp_new(:, j) = bp_j;
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Current consumption implied by updated policy

c = ygrid + r * bgrid - bp_new;

cmin3 = min(c(:));
if cmin3 <= 0
    error('egmUpdateSS:NonPositiveC', ...
        'Non-positive consumption implied by bp_new (min=%g). Check bounds / parameters.', cmin3);
end

