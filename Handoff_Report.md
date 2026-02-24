# Handoff Report: Heterogeneous Agent Model Simulation

This report summarizes the correction of the capital explosion issue and the current state of the codebase for further research/General Equilibrium (GE) tasks.

## 1. Current State of the Simulation Code
The simulation (Task 1.A) has been stabilized and corrected. 
- **Correction Method**: We identified that linear extrapolation of the policy function outside the grid $k \in [0, 10]$ was causing a divergence. The logic now implements **boundary clamping**: individuals' asset holdings are clamped to the grid bounds before querying the policy function.
- **Consistency**: Consumption is now derived directly from the budget constraint ($c = Rk + zw - k'$) after determining $k'$ from the policy function. This ensures that the simulated paths strictly obey the budget identities.
- **Statistical Robustness**: The stationary distribution statistics (Mean, Gini, Percentile Ratios) are now calculated by pooling all individuals across periods 401–1400, providing a more stable estimate of the invariant distribution.

## 2. Mathematical Notation
The model adheres to the standard Aiyagari-style notation:
- **State Variables**: Assets $k_t$ and productivity shock $z_t$.
- **Budget Constraint**: $C_t + k_{t+1} = R k_t + z_t w$, where $w$ is the wage.
- **Cash-on-Hand (for EGM)**: $x_t = R k_t + z_t w$.
- **Optimization**: $\max \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \frac{c_t^{1-\sigma}-1}{1-\sigma}$.
- **Constraints**: $k_{t+1} \geq 0$ (Non-negativity/Borrowing limit).

## 3. Global Parameters
These parameters are calibrated for Exercise 1 and must be maintained for consistency:

| Parameter | Symbol | Value | Description |
| :--- | :--- | :--- | :--- |
| **Discount Factor** | $\beta$ | `0.98` | Time preference |
| **Risk Aversion** | $\sigma$ | `2` | CRRA parameter |
| **Wage** | $w$ | `1` | Normalized efficiency wage |
| **Interest Rate** | $R$ | `0.995 / beta` | Approximately `1.0153` |
| **Persistence** | $\rho$ | `0.95` | AR(1) coefficient for $\log(z_t)$ |
| **Shock Variance** | $\sigma_\epsilon^2$ | `0.05` | Innovation variance for log-productivity |
| **Asset Grid** | `gri.k` | `[0, 10]` | 20 points, log-spaced |

## 4. Logic Transition References
In the main script `Ps1_Main_script.m`:
- **Simulation & Stats End**: Line **347** (finishes the Euler Equation Error check and reporting).
- **EGM Logic Begins**: Line **349** (initializes the grid for cash-on-hand `x`).
- **EGM Solver Call**: Line **363** (`solving_EGM = EGM(par, cpar, gri);`).

The code is now prepared for the General Equilibrium sweep (Question 2), where $R$ and $w$ will be endogenized via market clearing.
