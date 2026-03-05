# Problem Set 1: Handoff Report

## Context
This project is for **Quantitative Macroeconomics II - Problem Set 1**, which focuses on solving and simulating stationary equilibria in an economy with idiosyncratic risk and incomplete markets (Aiyagari model). The underlying methods implemented match those covered in class, utilizing tools like the Rouwenhorst discretization method, Value Function Iteration (VFI) with Golden Section search, and the Endogenous Grid Method (EGM).

## Code Architecture
The clean project directory contains the following primary file structure:
- **`0_codes/Ps1_Main.m`**: The master script that executes all configurations for the problem set. It is fully vectorized and computes solutions for Partial Equilibrium, General Equilibrium, Heterogeneous Returns, and includes extensive robustness checks. 
- **`0_codes/0_functions/`**: Contains helper functions heavily relied on by the main script:
  - `VFI_GS.m`: Solves the household problem using traditional Value Function Iteration paired with Golden Section Search.
  - `EGM.m`: Solves the household problem rapidly using the Endogenous Grid Method.
  - `EGM_het.m`: An adaptation of the EGM solver to handle an extended state space incorporating heterogeneous asset returns.
  - `rouwenhorst.m`: Discretizes highly persistent AR(1) processes (used for both productivity `z` and returns `r`).
  - `euler_errors.m`: Computes Euler equation errors to evaluate the accuracy of the solved policy functions.

## Implementation Details & Features
1. **Question 1 (Partial Equilibrium)**:
   - Sets up the household asset accumulation problem under strict borrowing limits.
   - Solves for optimal capital ($k'$) and consumption ($c$) policies via both **VFI** and **EGM**.
   - Simulates a panel of 2,000 households over 1,400 periods to approximate the stationary wealth distribution.
   - Computes summary statistics reflecting inequality, such as Gini coefficients and top wealth percentile ratios (90/10, 99/1).

2. **Question 2 (General Equilibrium)**:
   - Submits the EGM household solver to a bisection algorithm that iterates over the interest rate ($R$) to clear the aggregate capital market (solving for $K\_demand = K\_supply$).
   - Re-simulates households at the general equilibrium prices to determine the updated stationary wealth distribution.

3. **Question 3 (Heterogeneous Asset Returns)**:
   - Expands the standard state space to include a stochastic return shock $r \sim \text{AR(1)}$, introducing a 25-state joint transition matrix ($5 \times 5$ Kronecker product of $z$ and $r$).
   - Solves the model for two distinct return persistence parameters ($\rho_r = 0.0$ and $\rho_r = 0.9$).
   - Highlights the emergence of fat (Pareto) right-tails in the wealth distribution driven by persistent variation in asset returns.

4. **Robustness Checks** (Toggled via `run_robustness = true;` at the top of `Ps1_Main.m`):
   - **Extended k-Grid ($k_{max} = 80$, No Extrapolation)**: Resolves Q1 and Q2 to assess whether increasing the grid bounds alleviates boundary clumping without requiring extrapolation.
   - **Extra Configuration test**: Re-runs Q1, Q2, and Q3 with a modified variance calibration ($\sigma_e = 0.05$ directly), a much larger upper limit ($k_{max} = 100$ and $N_{kap} = 40$), and utilizes **extrapolation** across the board to prevent artificial pooling at the upper grid bounds. It handles plotting for the distributions and policy bounds accurately for this section.

## Status for the Next Agent
- All features requested across Q1, Q2, Q3, and the robustness checks have been fully implemented.
- Computations are heavily optimized via `griddedInterpolant` and vectorized simulations, reducing runtime drastically over grid-search VFI methods.
- Comprehensive plots of policies, Euler errors, and histograms layered with continuous Kernal Density Estimates (KDE) are automatically generated while iterating.
- **Action items moving forward**: The script is stable, and the next agent can run `0_codes/Ps1_Main.m` to pull the final deliverables (tables printed to terminal, high-resolution plots saved to `0_codes/1_graphs/`) straight into the final LaTex submission report.
