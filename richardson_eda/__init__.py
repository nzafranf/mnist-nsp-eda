"""Richardson Extrapolation EDA Module

Implements Algorithm 5.1 (Adaptive Δt via Richardson Extrapolation) from PROOF.md.
Provides tools for hyperparameter optimization and algorithm validation.
"""

from .adaptive_solver import AdaptiveFlowSolver, SolverConfig

__all__ = ['AdaptiveFlowSolver', 'SolverConfig']
