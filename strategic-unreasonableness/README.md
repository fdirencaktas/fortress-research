# Strategic Unreasonableness: A Tool of the Political Systematical Trader

This repository contains the complete Python code and simulation framework for the working paper *"Strategic Unreasonableness: A Tool of the Political Systematical Trader"* (Aktaş, 2026). The project tests whether introducing controlled, random deviation from a rational trading strategy can preserve profitability while becoming highly unpredictable to pattern-matching AI systems.

## Core Thesis

Against a pattern-learning AI adversary, any rule-based deviation strategy is eventually learnable. The only unlearnable behavior is true randomness. This repository empirically validates that 10-15% random deviation achieves near-maximum unpredictability (0.97) while preserving the baseline Sharpe ratio (0.35) on SPY data (2015-2025).

## Key Findings

- **Baseline (0% deviation)**: Sharpe = 0.35, Unpredictability = 0.00
- **Optimal (15% deviation)**: Sharpe = 0.35, Unpredictability = 0.97
- **Zero Sharpe loss for 97% unpredictability gain**


