# Handoff Report: General Equilibrium & Remaining Tasks

This report brings a new collaborator up to speed on the current state of `Ps1_Main_script.m`, the bugs that were fixed, the modelling choices that were made, and the remaining work for Questions 2 and 3.

---

## 1. Problem Set Overview

The problem set implements an **Aiyagari (1994)-style** incomplete-markets model. Agents face idiosyncratic productivity shocks $z_t$ and save in a single riskless asset $k_t$.

| Question | Description | Status |
| :--- | :--- | :--- |
| **Q1.A** | VFI with Golden-Section search | ✅ Complete |
| **Q1.A.3** | Simulation of 2000 individuals, 1400 periods | ✅ Complete |
| **Q1.A.4** | Euler equation errors (VFI) | ✅ Complete |
| **Q1.B** | EGM (Carroll 2006), policy comparison, Euler errors | ✅ Complete |
| **Q2.1** | Firm FOCs for capital demand and wage | ✅ Implemented |
| **Q2.2** | GE bisection on $R$ | ⚠️ Runs, needs verification |
| **Q2.3** | Wealth distribution histogram, Gini, percentile ratios | ⚠️ Code present, depends on Q2.2 |
| **Q3** | Heterogeneous returns ($\tilde{r}_t^i$) | ❌ Not started |

---

## 2. Bugs Found and Fixed

### 2.1 `interp_kp.m` — Catastrophic Penalty for Low Capital

**Original behaviour:** When $k' \leq 10^{-6}$, the function returned $-10^{10}$ instead of a proper interpolation. Since the asset grid starts at $k_1 = 0$, any agent near the borrowing constraint triggered this penalty, corrupting the entire simulation trajectory.

**Fix:** Replaced the branching logic with symmetric linear extrapolation on both tails. The file `0_functions/interp_kp.m` now has a clean structure with no ad-hoc penalties.

### 2.2 Simulation Shock Process — Variance vs. Standard Deviation

**Original code (lines 173, 382):**
```matlab
z_i(j, t+1) = exp(par.rho * log(z_i(j,t)) + cpar.sigmae * randn());
```
Since `cpar.sigmae = 0.05` is the **variance** $\sigma_\epsilon^2$, the innovation must be scaled by its square root. 

**Fix:** `sqrt(cpar.sigmae) * randn()`.

> [!NOTE]
> The first simulation (Q1.A.3) was later **fully rewritten** by the student to use a **discrete Markov chain** transition (via cumulative probabilities of `gri.prob`), which is the theoretically correct approach for simulating a discretised process. This fix is therefore only still relevant for the GE simulation block (lines 530–537), which still uses continuous AR(1) draws. See §4 for details.

### 2.3 Normalisation Drift in Continuous Simulation

When the Rouwenhorst grid is normalised so that $\mathbb{E}[z] = 1$, naïvely applying $\log(z_{i,t})$ in the AR(1) law introduces a systematic downward drift, because $\log(z)$ no longer has mean zero. The GE simulation block now tracks the raw log-process `log_z_i_GE` separately and normalises:
```matlab
log_z_i_GE(j, t+1) = par.rho * log_z_i_GE(j, t) + sqrt(cpar.sigmae) * randn();
z_i(j, t+1) = exp(log_z_i_GE(j, t+1)) / mean_z;
```

---

## 3. Key Modelling Decision: $MPK = R$ (Full Depreciation)

The PDF states explicitly that the marginal products of $K$ and $N$ equal $R$ and $w$ respectively. For a Cobb-Douglas production function $Y = K^\alpha N^{1-\alpha}$, this implies:

$$\alpha K^{\alpha - 1} N^{1-\alpha} = R$$

which is equivalent to assuming **full depreciation** ($\delta = 1$). This is intentional — it is a common simplification in problem sets. As a consequence:

- **Capital demand:** `k_d = (alpha / R)^(1/(1-alpha))`
- **Wage:** `w = (1-alpha) * (alpha / R)^(alpha/(1-alpha))`
- $R^*$ will be **below 1** (the gross return doesn't cover the full capital stock), which is normal under $\delta = 1$.

> [!CAUTION]
> An earlier revision mistakenly assumed $\delta = 0$ and used $MPK = R - 1$. This was reverted. The formulas above are now correct as per the PDF specification.

---

## 4. Current State of the GE Loop (Question 2.2)

The bisection loop lives at **lines 501–636** of `Ps1_Main_script.m`. Here is its logic:

```
R_min = 0.8,  R_max = 1/beta = 1.0204
while |K_supply - K_demand| > tol (1e-3)
    R_guess = (R_min + R_max) / 2
    K_demand = (alpha / R_guess)^(1/(1-alpha))
    w_guess  = (1-alpha) * (alpha / R_guess)^(alpha/(1-alpha))
    
    Rebuild capital grid: k_N = 10 * w_guess    ← adapts to new w
    Solve EGM(R_guess, w_guess)
    Simulate 2000 individuals × 300 periods using pre-drawn z shocks
    K_supply = mean(k_sim at t=300)
    
    if K_supply > K_demand → R_max = R_guess   (raise R to reduce savings)
    else                   → R_min = R_guess
```

### Known Issue: Simulation Method Mismatch

The **Q1.A.3 simulation** was rewritten to use **discrete Markov transitions** (cumulative probability lookup), which is consistent with the Rouwenhorst discretisation. However, the **GE simulation** (lines 530–605) still uses **continuous AR(1) draws** with `randn()`. This creates an inconsistency: the EGM policy was solved on the discrete $z$-grid, but the simulation evaluates it at arbitrary continuous $z$ values via interpolation. This can affect convergence quality and equilibrium precision.

> [!IMPORTANT]
> **Recommendation:** Port the discrete Markov simulation approach from Q1.A.3 into the GE loop for consistency. Pre-draw the `z_idx` matrix once before the bisection, then inside the loop only simulate $k$ forward using `interp1` on the policy function by productivity state (as in lines 194–224).

---

## 5. Global Parameters

| Parameter | Symbol | Value | Notes |
| :--- | :--- | :--- | :--- |
| Discount Factor | $\beta$ | `0.98` | |
| Risk Aversion | $\sigma$ | `2` | CRRA |
| AR(1) Persistence | $\rho$ | `0.95` | For $\log(z_t)$ |
| Innovation Variance | $\sigma_\epsilon^2$ | `0.05` | Use `sqrt(0.05)` for std dev |
| Borrowing Limit | $b$ | `0` | $k_{t+1} \geq 0$ |
| Capital Share | $\alpha$ | `0.33` | Not given in PDF; assumed |
| Rouwenhorst States | $N$ | `5` | |
| Capital Grid | $k$ | `[0, 10w]` | 20 log-spaced points; updated each GE iter |
| Partial Eq. Rate | $R$ | `0.995/\beta ≈ 1.0153$ | Used in Q1 only |

---

## 6. Code Map

| Lines | Section | Key Variables |
| :--- | :--- | :--- |
| 1–41 | Parameters & Rouwenhorst discretisation | `par`, `cpar`, `gri.z`, `gri.prob`, `mean_z` |
| 43–52 | Capital grid setup | `gri.k` (20 log-spaced points) |
| 54–138 | **Q1.A** — VFI + policy/value plots | `V`, `cpol`, `kpol` (all `Nz × Nk`) |
| 140–283 | **Q1.A.3** — Discrete simulation + wealth stats | `z_idx`, `k_i`, `k_vec`, Gini, ratios |
| 284–347 | **Q1.A.4** — VFI Euler errors | `EE` (`Nz × Nk`) |
| 349–498 | **Q1.B** — EGM, comparison plots, EGM Euler errors | `solving_EGM`, `kpol_EGM`, `EE_EGM` |
| 501–636 | **Q2** — GE bisection loop | `R_guess`, `k_d`, `w_guess`, `k_supply` |
| 644–691 | **Q2.3** — GE wealth distribution & inequality stats | histogram, Gini, 90/10, 99/1 |

---

## 7. Remaining Work: Question 3

Question 3 introduces **heterogeneous asset returns**:
$$R_t^i = 1 + \bar{r} + \tilde{r}_t^i$$
where $\tilde{r}_t$ follows an independent AR(1) with persistence $\rho_r$ and innovation variance $\sigma_r^2 = (1 - \rho_r^2) \cdot 0.002$.

### Sub-tasks

1. **Q3.1** — Discretise the return process (Rouwenhorst, 5 states). Construct a **25×25** joint transition matrix for $(z, r)$. Solve the consumer's problem for a given $\bar{r}$ using EGM (or VFI). Plot policies for the highest and lowest return states.
2. **Q3.2** — Embed in a GE loop: simulate the economy, bisect on $\bar{r}$ to clear the capital market.
3. **Q3.3** — Wealth distribution analysis: Gini, percentile ratios for both $\rho_r = 0$ and $\rho_r = 0.9$. Test for Pareto tail by plotting $\log(1 - \hat{F}(k))$ vs $\log(k)$.

### Q3 Parameters

| Parameter | Value |
| :--- | :--- |
| $\bar{r}$ | `0.985/beta - 1` (to be endogenised) |
| $\rho_r$ | Case 1: `0`, Case 2: `0.9` |
| $\sigma_r^2$ | `(1 - rho_r^2) * 0.002` |
| Return grid | 5 states via Rouwenhorst |

### Implementation Notes

- The state space expands from $(k, z)$ to $(k, z, r)$. With 5 states each for $z$ and $r$, the joint exogenous state has **25** nodes.
- The joint transition matrix is the **Kronecker product** $P_{joint} = P_z \otimes P_r$ (since $z$ and $r$ are independent).
- The EGM Euler equation becomes $R$-state-dependent: $c_t = u'^{-1}\{\beta \mathbb{E}[R'_{i} \cdot u'(c_{t+1}) \mid z, r]\}$.
- The capital grid upper bound may need to increase substantially, as heterogeneous returns can generate fat-tailed wealth distributions.

---

## 8. File Structure

```
0_codes/
├── Ps1_Main_script.m          ← Main driver (Questions 1–2)
├── Ps1_Ferrari_Jimenez.m      ← (Unused / earlier version)
├── Handoff_Report.md          ← Previous handoff (Q1.A stabilisation)
├── Handoff_Report_GE.md       ← This report
├── PS 1 no labor final.pdf    ← Problem set specification
└── 0_functions/
    ├── bellman_RHS.m           ← Bellman objective for VFI
    ├── EErrors.m               ← (Euler error helper)
    ├── EGM.m                   ← Endogenous Grid-point Method solver
    ├── interp_kp.m             ← Linear interpolation/extrapolation
    ├── k_motion.m              ← Capital law of motion
    ├── MaxGoldenSearch.m       ← Golden-section search
    ├── rouwenhorst.m           ← Rouwenhorst discretisation
    ├── u.m                     ← CRRA utility
    └── VFI_GS.m                ← Value Function Iteration
```
