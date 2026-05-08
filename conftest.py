"""
Root conftest.py — applies a fast Hypothesis profile for the entire test suite.

The 'fast' profile caps max_examples at 20 so property-based tests complete
quickly during development. Increase to 200+ for thorough CI runs by setting
the HYPOTHESIS_PROFILE environment variable or overriding per-test with
@settings(max_examples=N).
"""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "fast",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

settings.load_profile("fast")
