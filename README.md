# Quantitative Macroeconomics 2 — Problem Set 1

This project implements the Aiyagari (1994) incomplete-markets model without labor supply. It includes:
1. Endogenous Grid Method (EGM) and Value Function Iteration (VFI) solvers
2. General Equilibrium bisection to find the market-clearing interest rate
3. An extension with heterogeneous idiosyncratic asset returns (Q3)

## Folder Structure

- `0_codes/Ps1_Main.m`: The main executable script that runs all questions, outputs tables, and generates figures.
- `0_codes/0_functions/`: Contains the core solver algorithms (`EGM`, `VFI_GS`) and helper routines (`rouwenhorst`, `euler_errors`).
- `1_graphs/`: Auto-generated directory where all plots and figures are saved.
- `PS1_Holistic_Handoff.md`: Model definition, calibration values, and known mathematical pitfalls.

## Usage

Simply run `0_codes/Ps1_Main.m` in MATLAB. The script is structured sequentially:
1. Question 1 (Partial Equilibrium using VFI and EGM)
2. Question 2 (General Equilibrium Bisection)
3. Question 3 (Heterogeneous Returns Extension)

By default, the script generates comprehensive comparative tables in the MATLAB console and saves all relevant histograms and policy function plots to the `1_graphs/` directory.

You can strictly toggle the robustness checks (e.g. testing `k_max = 80`) by setting the `run_robustness` flag at the top of the main script.
