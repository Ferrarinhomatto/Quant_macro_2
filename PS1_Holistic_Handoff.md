# PS1 — Aiyagari Model: Holistic Handoff Report

*Quantitative Macroeconomics 2 — March 2025*

> **PURPOSE**: This report describes the economic model, numerical methods, equations, and known pitfalls for PS1. It is deliberately **code-agnostic** — filenames and variable names in any implementation may differ. Use it as context when reviewing code in any folder.

---

## 1. Model

Standard **Aiyagari (1994)** incomplete-markets economy *without* labor supply:

$$\max\;\mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \frac{c_t^{1-\sigma}-1}{1-\sigma}$$

subject to:

$$c_t + k_{t+1} = R\,k_t + z_t\,w, \qquad k_{t+1} \geq 0$$

- **CRRA utility** with σ = 2.
- Idiosyncratic labor productivity z follows log(z) ~ AR(1) with ρ = 0.95, σ²_ε = 0.05.
- Discretised using **Rouwenhorst** with N_z = 5 states.
- The support z is **exponentiated** and **mean-normalised** to E[z] = 1 using the ergodic distribution.

---

## 2. Calibration

| Parameter | Symbol | Value | Notes |
|:---|:---|:---|:---|
| Discount factor | β | 0.98 | |
| Risk aversion | σ | 2 | CRRA |
| AR(1) persistence | ρ | 0.95 | For log z |
| Innovation variance | σ²_ε | 0.05 | Pass √0.05 to Rouwenhorst |
| Borrowing limit | b | 0 | k' ≥ 0 |
| Capital share | α | 0.33 | |
| Depreciation | δ | 0.025 | Partial depreciation |
| Grid points | N_k | 20 | Log-spaced on [0, 10] |
| Rouwenhorst states | N_z | 5 | |

### Partial-equilibrium interest rate (Q1 only)

R_PE = 0.995 / β ≈ 1.0153

This ensures βR < 1 (required for a stationary wealth distribution).

---

## 3. Question-by-Question Specification

### Q1.A — Value Function Iteration (Golden-Section Search)

**Goal**: Solve the household Bellman equation in partial equilibrium.

**Bellman equation**:

V(k, z) = max_{c} { u(c) + β Σ_{z'} π(z'|z) V(k', z') }

where k' = Rk + zw - c, and c ∈ [ε, Rk + zw].

**Method**: For each (z_j, k_i), maximize over c using **golden-section search** on a continuous interval. The continuation value V(k', z') is obtained by **linear interpolation** on the k-grid.

**Outputs**: Value function V(z, k), consumption policy c(z, k), savings policy k'(z, k).

**Matrix convention**: VFI typically stores (N_z × N_k).

**Deliverables**: Plots of c(k,z), k'(k,z), and V(k,z) for all 5 states.

---

### Q1.A.3 — Simulation

**Goal**: Simulate 2000 individuals for 1400 periods, discard first 400 as burn-in.

**Report**: Wealth distribution histogram, Gini coefficient, 90/10 and 99/1 percentile ratios.

**Method**:
1. Pre-draw shocks: Pre-simulate the discrete Markov chain for all agents/periods at once.
2. For each period: Interpolate k'(k, z) on the grid for each productivity state.
3. Borrowing constraint: Enforce k' ≥ 0 after interpolation.

**Pitfall**: Linear extrapolation beyond the grid can cause capital explosion. Two valid approaches:
- **Clamp** queries to [k_min, k_max] (conservative, slight bias at upper bound)
- **Extrapolate** linearly but enforce k ≥ 0 only; expand grid if agents pile up at boundary

---

### Q1.A.4 — Euler Equation Errors (VFI)

**Euler equation** (interior points where k' > 0):

c^(-σ) = βR Σ_{z'} π(z'|z) c(k', z')^(-σ)

**Euler-implied consumption**:

c_Euler = (βR Σ_{z'} π(z'|z) c(k', z')^(-σ))^(-1/σ)

**Error metric** (absolute percentage):

EE(z, k) = |c_Euler - c_VFI| / c_VFI × 100

**CRITICAL**: Borrowing-constrained points (k' = 0) must be **excluded** from the max-error statistic. At these points the Euler equation holds as an **inequality**, so the "error" is by design.

**Expected accuracy** with 20 grid points: ~0.5–3% at interior points (log₁₀ ≈ -1.5 to -2).

---

### Q1.B — Endogenous Grid-Point Method (EGM)

**Goal**: Solve the same problem using Carroll (2006) EGM, compare with VFI.

**Method**:
1. Grid on next-period capital k'.
2. For each k' and current z: compute expected marginal utility
   EMU = Σ_{z'} π(z'|z) c(x', z')^(-σ)
   where x' = z'w + Rk' is tomorrow's cash-on-hand.
3. Invert the Euler equation: c_today = (βR · EMU)^(-1/σ)
4. Compute endogenous cash-on-hand: x_end = k' + c_today
5. Invert the mapping from (x_end, k') → (x_exogenous, k').

**Borrowing constraint**: Prepend the point (x=0, k'=0) before inverting.

**Matrix convention**: EGM typically stores (N_k × N_z) — **transposed** relative to VFI.

**Deliverables**:
- Overlay plots: VFI (solid) vs EGM (dashed)
- Sup-norm and L2-norm of policy differences
- EGM Euler errors (should be near machine precision, ~10^(-12)%)

---

### Q2 — General Equilibrium (Bisection on R)

**Goal**: Find equilibrium R* that clears the capital market.

**Firm FOCs** (Cobb-Douglas, partial depreciation):

R = 1 + αK^(α-1) - δ,    w = (1-α)K^α

Inverting: K_d = (α / (R - 1 + δ))^(1/(1-α))

**Bisection**:
1. Bounds: R_min = 0.995, R_max = 1/β.
2. Guess R = midpoint. Compute K_d, w from firm FOCs.
3. Solve EGM → simulate 2000 agents × 300 periods → K_s = mean(k_terminal).
4. If K_s > K_d: raise R_min. Else: lower R_max.
5. Converge when |K_s - K_d| < 10^(-3).

**Key decisions**:
- Capital grid [0, 10] should be **fixed** across iterations.
- Warm-start agents from previous iteration's terminal capital.
- Expect R* ≈ 1.01–1.02 with δ = 0.025, β = 0.98.

**Deliverables**: R*, w*, K*, wealth distribution, Gini, 90/10, 99/1.

---

### Q3 — Heterogeneous Asset Returns

**Extension**: Each agent receives an idiosyncratic return:

R^i_t = 1 + r̄ + r̃^i_t

where r̃ follows an independent AR(1):

r̃_{t+1} = ρ_r r̃_t + ε_t,    Var(ε) = (1 - ρ²_r) × 0.002

This ensures unconditional Var(r̃) = 0.002 regardless of ρ_r.

**Two cases**: ρ_r = 0 (i.i.d.) and ρ_r = 0.9 (persistent).

#### Q3.1 — Joint state space

Discretise r̃ with Rouwenhorst (N_r = 5). **Pass √σ²_r, not σ²_r**.

Joint state: s = (z_idx - 1) × N_r + r_idx → 25 states.

Joint transition: P_joint = kron(P_z, P_r) — z outer, r inner.

#### Q3.1 — EGM with state-dependent returns

**KEY DIFFERENCE**: R is now **inside** the expectation:

Standard (Q1/Q2): c_today = (βR · EMU)^(-1/σ)       ← R scalar, outside
Q3 heterogeneous: c_today = (β · EMU_het)^(-1/σ)     ← R' folded into EMU

where EMU_het = Σ_{s'} π(s'|s) R(s') c(x', s')^(-σ)

**WARNING**: Pulling R out of the expectation kills the precautionary savings from return risk.

#### Q3.2 — GE Bisection on r̄

Bisect on r̄ (not R). Firm FOCs: r_k = r̄ + δ, K_d = (α/r_k)^(1/(1-α)), w = (1-α)K^α_d.

Simulate both z and r̃ shocks independently, combine into joint index.

#### Q3.3 — Wealth distribution analysis

For each ρ_r case: Gini, 90/10, 99/1, Pareto tail test (log(1-CDF) vs log(k)).

**Expected results**:
- ρ_r = 0.9 → **higher Gini**, **fatter right tail** (closer to Pareto)
- ρ_r = 0 → lower inequality, thinner tail (near-exponential)

---

## 4. Key Equations Summary

| Equation | Formula |
|:---|:---|
| Budget constraint | k' = Rk + zw - c |
| CRRA utility | u(c) = (c^(1-σ) - 1)/(1-σ) |
| Marginal utility | u'(c) = c^(-σ) |
| Euler (R scalar) | c^(-σ) = βR Σ π(z'\|z) c(k',z')^(-σ) |
| Euler (R state-dep.) | c^(-σ) = β Σ π(s'\|s) R(s') c(k',s')^(-σ) |
| Firm FOC: R | R = 1 + αK^(α-1) - δ |
| Firm FOC: w | w = (1-α)K^α |
| Gini (discrete) | G = 2Σ(i·k_(i)) / (N·Σk_(i)) - (N+1)/N |

---

## 5. Pitfalls Table

| # | Pitfall | Consequence | Fix |
|:---|:---|:---|:---|
| 1 | Pass σ² instead of √σ² to Rouwenhorst | Grid too wide/narrow | Always pass **std dev of innovations** |
| 2 | Use δ=1 instead of δ=0.025 | R* < 1, wrong economics | δ = 0.025 gives R* > 1 |
| 3 | Rebuild k-grid each GE iteration | Bisection unstable | Fix the grid once |
| 4 | Don't normalise z to mean 1 | Aggregate labor ≠ 1 | Divide by π'z after exponentiating |
| 5 | Include constrained points in Euler error max | Inflated error | Exclude points where k' ≤ 0 |
| 6 | Pull R outside expectation in Q3 | Wrong precautionary savings | Use β·EMU_het, not βR·EMU |
| 7 | Clamp vs extrapolate k in simulation | Bias vs explosion | Extrapolate + enforce k≥0; widen grid if needed |
| 8 | VFI/EGM matrix convention mismatch | Index errors | VFI = (N_z × N_k), EGM = (N_k × N_z) — transpose |
| 9 | Wrong Kronecker order | Wrong joint transition | kron(P_z, P_r): s = (z-1)N_r + r |

---

## 6. Code Review Checklist

- [ ] Rouwenhorst: √σ² passed, not σ²?
- [ ] z normalised to E[z] = 1?
- [ ] δ = 0.025, not 1?
- [ ] VFI golden-section correctly evaluates Bellman with interpolated continuation values?
- [ ] EGM inverts endogenous grid correctly with borrowing constraint prepended?
- [ ] EGM uses c = (βR · EMU)^(-1/σ)?
- [ ] Euler errors exclude constrained points (k' = 0)?
- [ ] GE bisection holds k-grid fixed? Firm FOCs correct?
- [ ] Q3 EGM_het: R **inside** expectation: c = (β · EMU_het)^(-1/σ)?
- [ ] Q3 Kronecker: kron(P_z, P_r) with s = (z-1)N_r + r?
- [ ] Q3 bisection on r̄ with r_k = r̄ + δ?
- [ ] Q3 shocks: z and r̃ simulated independently, different seeds?
- [ ] Q3 results: ρ_r=0.9 → more inequality than ρ_r=0?
