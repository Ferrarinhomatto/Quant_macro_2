function gNext = gUpdate(g, bp, bgrid, f, sigma)
%GUPDATE  One Young (2010)-style non-stochastic distribution update.
%
%   gNext = gUpdate(g, bp, bgrid, f, sigma)
%
% Inputs
%   g     : (Nb x 2) current distribution; g(:,j) is mass over assets in state j
%             (j=1 employed, j=2 unemployed)
%   bp    : (Nb x 2) saving policy b'(b,x); bp(i,j) = b'(bgrid(i), x=j),
%             clipped to [bgrid(1), bgrid(end)]
%   bgrid : (Nb x 1) asset grid (strictly increasing)
%   f     : scalar job-finding probability (U->E)
%   sigma : scalar separation probability (E->U)
%
% Output
%   gNext : (Nb x 2) distribution one period forward
%
% Method:
%   1) For each current state j, split mass along assets using bp(:,j)
%      via linear interpolation onto the two neighbouring grid points.
%   2) Apply the employment transition (implied by f and sigma) to map
%      mass across next-period employment states.
%   Total mass is preserved up to numerical rounding; tiny drift is renormalised.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0) Input checks / coercions
% NOTE: This section has some very strict checks to make sure you
% don't input incorrect things into the function. 

if nargin ~= 5
    error('gUpdate:BadNargin', 'Expected 5 inputs: g, bp, bgrid, f, sigma.');
end

% bgrid
if ~isnumeric(bgrid) || ~isvector(bgrid) || isempty(bgrid)
    error('gUpdate:BadBgrid', 'bgrid must be a nonempty numeric vector.');
end
bgrid = double(bgrid(:));
Nb = numel(bgrid);
if any(~isfinite(bgrid))
    error('gUpdate:BadBgrid', 'bgrid contains non-finite values.');
end
if any(diff(bgrid) <= 0)
    error('gUpdate:NonMonotoneGrid', 'bgrid must be strictly increasing.');
end
bmin = bgrid(1);
bmax = bgrid(end);

% Scalars f, sigma
if ~isnumeric(f) || ~isscalar(f) || ~isfinite(f)
    error('gUpdate:BadF', 'f must be a finite numeric scalar.');
end
if ~isnumeric(sigma) || ~isscalar(sigma) || ~isfinite(sigma)
    error('gUpdate:BadSigma', 'sigma must be a finite numeric scalar.');
end
f = double(f);
sigma = double(sigma);
if f < 0 || f > 1
    error('gUpdate:BadFRange', 'f must be in [0,1].');
end
if sigma < 0 || sigma > 1
    error('gUpdate:BadSigmaRange', 'sigma must be in [0,1].');
end

% g
if ~isnumeric(g) || ~isreal(g) || any(~isfinite(g(:)))
    error('gUpdate:BadG', 'g must be a real, finite numeric array.');
end
g = double(g);

% bp
if ~isnumeric(bp) || ~isreal(bp) || any(~isfinite(bp(:)))
    error('gUpdate:BadBP', 'bp must be a real, finite numeric array.');
end
bp = double(bp);

% Determine Nx from g and bp; should be 2 (E,U)
if ~ismatrix(g) || ~ismatrix(bp)
    error('gUpdate:BadDims', 'g and bp must be 2D arrays.');
end

% Accept transposed shapes
if size(g,1) ~= Nb && size(g,2) == Nb
    g = g.'; % try to coerce to Nb-by-Nx
end
if size(bp,1) ~= Nb && size(bp,2) == Nb
    bp = bp.'; % try to coerce to Nb-by-Nx
end

if size(g,1) ~= Nb
    error('gUpdate:BadGSize', 'g must have Nb=%d rows (grid size).', Nb);
end
if size(bp,1) ~= Nb
    error('gUpdate:BadBPSize', 'bp must have Nb=%d rows (grid size).', Nb);
end
Nx = size(g,2);
if size(bp,2) ~= Nx
    error('gUpdate:StateDimMismatch', 'bp must have same #columns as g (Nx=%d).', Nx);
end
if Nx ~= 2
    error('gUpdate:BadNx', 'This PS expects Nx=2 states (E,U). Got Nx=%d.', Nx);
end

% Distribution sanity
if any(g(:) < -1e-14)
    error('gUpdate:NegativeMass', 'g contains negative entries (min=%g).', min(g(:)));
end
g(g < 0) = 0; % clip tiny negatives

% Policy bounds
tol = 1e-12 * max(1, max(abs(bgrid)));
if any(bp(:) < bmin - tol) || any(bp(:) > bmax + tol)
    badLo = sum(bp(:) < bmin - tol);
    badHi = sum(bp(:) > bmax + tol);
    error('gUpdate:BPOutOfBounds', ...
        'bp must lie within [bmin,bmax]. Found %d below and %d above.', badLo, badHi);
end
bp = min(max(bp, bmin), bmax);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Build transition matrix Pi

% Ordering: 1=E, 2=U
Pi = [1 - sigma, sigma;
    f,         1 - f];


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Update

gNext = zeros(Nb, Nx);
mass0 = sum(g(:));

for j = 1:Nx
    gj  = g(:, j);
    bpj = bp(:, j);

    % Locate bpj on grid: bgrid(ell) <= bpj <= bgrid(ell+1)
    [~, ell] = histc(bpj, bgrid);   %#ok<HISTC>  % NOTE: histc has been deprecated 
                                                 % and this is not best
                                                 % approach

    % Safety: should not happen after clamp
    ell(ell == 0) = 1;

    % If bpj == bmax, histc returns ell=Nb -> we assign all mass to Nb
    atTop = (ell == Nb);
    ell(atTop) = Nb - 1;  % so ell+1 is valid
    ellp1 = ell + 1;

    denom = bgrid(ellp1) - bgrid(ell);
    if any(denom <= 0)
        error('gUpdate:BadDenom', 'Non-positive grid spacing encountered.');
    end

    omega = (bgrid(ellp1) - bpj) ./ denom; % weight on ell
    omega = min(max(omega, 0), 1);
    omega(atTop) = 0; % if atTop, put everything on ellp1 (=Nb)

    mLow  = omega .* gj;
    mHigh = (1 - omega) .* gj;

    % First: deposit mass onto asset grid for each current state j
    % (we’ll then apply Pi to allocate across next-state jp)
    % But it's cheaper to apply Pi immediately:
    for jp = 1:Nx
        w = Pi(j, jp);
        if w ~= 0
            gNext(:, jp) = gNext(:, jp) + ...
                accumarray(ell,   w*mLow,  [Nb,1], @sum, 0) + ...
                accumarray(ellp1, w*mHigh, [Nb,1], @sum, 0);
        end
    end
end

% Clean up numerical noise
gNext(gNext < 0) = 0;

% Optional: renormalise if tiny drift
mass1 = sum(gNext(:));
if mass0 > 0 && mass1 > 0
    if abs(mass1 - mass0) > 1e-12 * max(1, mass0)
        gNext = gNext * (mass0 / mass1);
    end
end

