# Single-Measurement Quantum Kernels Achieve Perfect Classification beyond Classical Limits

This repository accompanies our NeurIPS 2024 submission.

## Summary

Classical kernel methods are fundamentally limited in their ability to learn certain highly nonlocal Boolean functions, such as Parity, Hidden Shift, and Forrelation. In this work, we show that quantum kernel methods—when equipped with appropriate quantum feature maps and even restricted to a single measurement—can achieve **perfect classification** on these problems, even when all classical polynomial kernels provably fail.

Our experiments systematically compare quantum kernel SVMs (using parity-based feature maps) to classical polynomial kernel SVMs across three key global Boolean function benchmarks:
- **Parity**: Classification based on the parity of input bits.
- **Hidden Shift**: Classification based on parity after a secret bitwise shift.
- **Forrelation**: Classification based on forrelated Boolean oracles.

In all cases, quantum kernel SVMs achieve perfect or near-perfect accuracy (even in the presence of limited training data), while classical polynomial kernels only succeed for trivial cases or fail completely.

These results provide the first concrete, simulation-based evidence of a strict quantum advantage in kernel learning, under physically realistic and resource-constrained conditions (i.e., a single quantum measurement).

For full details, please refer to our manuscript.
