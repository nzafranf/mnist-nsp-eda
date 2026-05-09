"""
Jacobian EDA Suite - Linearizability Analysis for Flow Matching

Computes geodesic deviation integrals to identify which time intervals
are "straightenable" (can use larger ODE steps without accuracy loss).

Main components:
- utils/jacobian_utils.py: Core Jacobian and integration routines
- scripts/compute_jacobian_eda.py: Main analysis pipeline
- tests/test_jacobian_pipeline.py: Test suite with dummy data
"""

__version__ = "1.0.0"
__author__ = "Flow Matching Research"
