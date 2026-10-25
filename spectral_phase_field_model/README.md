# Electrochemistry Coupled Phase Field  FFT Solver (Allen-Cahn, Cahn-Hilliard, Mechanics and Electrochemistry) in Lithium-ion Solid State Battery

## r1  J2 Plasticity 

J2 plasticity is unsuitable for polymers because it ignores pressure-dependence, volume changes, and viscoelasticity.
However, in the context of this specific spectral phase-field code, J2 is used as a computational necessity. It provides a mathematically elegant, computationally cheap way to regularize infinite stress singularities at the dendrite tip and acts as a phenomenological proxy for local polymer creep/stress-relaxation, allowing the FFT solver to run efficiently while still capturing the primary physics of dendrite propagation along grain boundaries.
The provided code uses a Spectral (FFT) Solver. FFT solvers are blazingly fast (O(N log N)) because they solve linear elasticity and simple J2 plasticity via algebraic operations in Fourier space (convolutions).
J2 (von Mises) Flow Rule : J2 perfectly captures metal dislocation glide and creep. It ignores hydrostatic pressure, and this does not produce error for metals.



## r1: a single set of VAR for the entire grains within the two  phase states (electrode and electrolyte). the MPF order parameters are represented by the name "ETA"

[![continuummodelnt2d](https://img.shields.io/badge/onevar1-streamlit-blue)](https://lithium-dendrites-in-solid-state-battery4.streamlit.app/) ( J2 flow rule or J2 plasticity for both Li metal and polymers,  All grains within electrode or electrolyte  are represented by a MPF order parameters, Temperature slider to model realistic temperature conditions, )
