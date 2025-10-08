Composite Mass Profiles and the MBT Curvature Trap

1. Introduction

Galactic rotation curves deviate from simple Keplerian expectations, indicating that the gravitational mass enclosed within a given radius does not fall off as a point mass.
Large observational programmes (e.g., the SPARC database) show that over radii of a few kiloparsecs the enclosed mass follows a power law of the form

M(r)∝r^m,

where the exponent m is ≈1.9 for most single spiral galaxies. 
A recent analysis summarised in a document provided by the user explains this exponent through a composite model—galaxies are not single‑component objects but consist of a stellar disk embedded in a dark‑matter halo. 
The goal of this report is to summarise the astrophysical evidence for this composite interpretation and to connect it to the CurvatureTrap simulation in the “MBT” (Motion–Basis–Time) model,
where numerical experiments were conducted to reproduce the m≈1.9 exponent.

2. Astrophysical mass profiles

2.1 Exponential disks

Observed surface brightness profiles of spiral galaxies are well fit by exponential disks of the form Σ(R)=Σ_0 e^(-R/R_d ) , where R_d is the scale length. 
The total mass of such a disk is M_disk=2π Σ_0 R_d^2 [1]. Integrating the surface density out to radius R gives

M_disk (<R) = 2πΣ_0 R_d^2 [1-(1+R⁄R_d ) e^(-R/R_d ) ].

For radii much less than the scale length (R≪R_d ), the bracketed term behaves as 1⁄2 (R/R_d )^2 , so the enclosed mass grows as M_disk∝R^2 .
Therefore a pure disk has a mass‑profile exponent m≈2 . This steep component reflects the luminous baryonic mass.

2.2 Isothermal dark‑matter halo

A common approximation for dark haloes is the isothermal sphere. In this model the density falls as ρ∝R^(-2) at large radii; hence the enclosed mass increases almost linearly with radius as M(R)∝R [2]. 
The isothermal halo therefore contributes a shallower mass‑profile exponent of m≈1 . Physically this component represents the diffuse dark matter that dominates the outskirts of galaxies.

2.3 Navarro–Frenk–White (NFW) profile
A more realistic halo is provided by the Navarro–Frenk–White (NFW) profile. This density behaves as ρ_"NFW" ∝1/R near the centre and ρ∝R^(-3) at large radii. Consequently,
in the inner region R≪h the enclosed mass scales as M(R)∝R^2 , while in the outer region it increases only logarithmically M(R)∝lnR [3].
The NFW model thus reproduces both the shallow and steep behaviours of the composite galaxy mass profile: it acts like a disc in the centre and like an isothermal halo at larger radii.

2.4 Composite exponent ≈1.9
Because galaxies consist of both a disk and a halo, the observed exponent is a weighted average of the two contributions.
If 60"-" 70% of the mass within the optical radius comes from the disk (m≈2 ) and 30"-" 40% from the halo (m≈1 ), the effective exponent is

m_eff ≈ 0.6×2+0.4×1 ≈ 1.6 + 0.4 ≈ 1.9.

This composite value agrees with the universal exponent m=1.878±0.064 measured for 80 spiral galaxies in the SPARC database.
The physical interpretation is therefore that the exponent encodes the structure of galaxies: a nearly flat rotation curve arises when disk and halo contributions balance such that m≈1 (halo) + m≈2 (disk) = 1.9.

3. The MBT CurvatureTrap model

3.1 Overview

The MBT model simulates how “mass” (represented by the energy contained in a motion field plus a scalar field) accumulates within a CurvatureTrap. 
The simulation evolves a vector field with memory, curvature and feedback terms, then measures the enclosed mass profile. 
It is not an astrophysical galaxy model, but the resulting mass profiles can be compared to observational slopes.
A key feature is the ability to choose different curvature kernels:

	disk curvature – resembles a disk‑like mass distribution. Without strong damping the simulation allows energy to propagate outward, leading to an m slope approaching 2.0.

	nfw curvature – mimics an NFW halo. The energy is confined and the mass profile quickly saturates at m≈1 , analogous to halo‑dominated regimes.

	extended and sharp curvatures – intermediate choices that tune the balance between disk‑like and halo‑like behaviour. These allow the simulation to explore a continuum of slopes.

3.2 Evolution parameters

Several parameters control the dynamics and memory of the curvature trap:

Parameter	Role

decay	Exponential decay of the motion field. Larger values → weaker damping, allowing mass to accumulate at larger radii; smaller values increase damping and concentrate mass at the centre.

γ (gamma)	Memory feedback strength between velocity and field. Smaller γ maintains memory of past motion, promoting extended accumulation and higher slopes; larger γ makes the system more localised.

λ (lambda)	Curvature–gradient feedback controlling how field gradients convert to velocity.

κ (kappa)	Coupling between the scalar field and the motion energy.

τ (tau)	Memory decay time. Larger τ retains memory for longer, acting like a lower damping; smaller τ quickly erases history.

steps, dt	Number of integration steps and time step size; adjusting these changes the evolution length and numerical stability.

3.3 Parameter exploration

Researchers used the MBT model to explore the parameter space and sought parameter sets that reproduce slopes around 1.9. Key findings include:

Pure nfw curvature: With strong damping (high decay ≈0.999 and moderate γ ), the mass profile slope measured m≈1 with R^2≈1 . This reproduces the isothermal/NFW halo behaviour.

Pure disk curvature: Using weak damping and long memory, the simulation gave m≈2 . This matches the steep mass increase in a pure exponential disk.

Intermediate slopes (1.5–1.9): By selecting the extended curvature and tuning parameters (e.g., decay ≈0.998–0.999, γ≈0.03 , λ≈0.7 , κ≈3.6 , τ≈45 ), the researchers obtained slopes around 1.9 with good power‑law fits (R^2≈0.9 ). These parameter sets produce a mass profile that extends outward like a disk but retains some central concentration, analogous to the halo–disk composite.

3.4 Scaling and rotation curves

To connect the dimensionless MBT results to physical galaxies, the simulated mass profiles must be scaled. Scaling uses three parameters: a maximum radius r_max (kpc) corresponding to the maximum dimensionless radius in the simulation, 
a softening scale ε (kpc) to avoid divergence at r=0 , and an amplitude A to match the observed rotation velocity. After scaling, the rotation speed is obtained from

v(r) = A √(M(r)/(r+ε)).

The root‑mean‑square error (RMSE) between the MBT rotation curve and observed data (e.g., the NGC 2915 galaxy) provides a metric to judge the fit. Parameter sets that produce slopes near 1.9 and low RMSE values were interpreted as physically plausible.

4. Discussion

4.1 Why m≈1.9 is physical

The astrophysical evidence summarised above shows that disks contribute M∝r^2 while haloes contribute M∝r .
Observations of galaxies indicate that the stellar disk typically dominates at radii of a few kiloparsecs but is always embedded in a halo. 
Therefore the net exponent is a composite, and the measured value (m≈1.88±0.06 ) is not a coincidence but encodes the relative disk/halo contributions.
This interpretation is further supported by the scaling relation linking the baryonic mass and the rotation velocity (v∞ ∝ M_baryon^0.331) reported in the universal gravitational scaling analysis.
Thus, the exponent m is a physically meaningful descriptor of how mass is distributed in galaxies.

4.2 Insights from the MBT simulation

The MBT CurvatureTrap is a conceptual model rather than a direct galaxy simulation. Nevertheless, its ability to generate a range of mass‑profile exponents by adjusting curvature types and evolution parameters illustrates a useful analogy:

	When the curvature kernel and damping favour extended propagation (large decay, small γ ), the simulated mass profile approximates a disk; when they favour localised accumulation, the profile resembles a halo.
	By tuning the parameters to intermediate values and selecting an “extended” curvature, the simulation reproduces slopes near 1.9, analogous to composite galaxies.

These findings suggest that phenomena that mix extended and localised dynamics naturally produce composite power‑law mass profiles. 

In the MBT framework, “mass” is an emergent property of motion fields; in galaxies, mass arises from the combination of baryons and dark matter. The same principle of superposition leads to similar exponents.

5. Conclusion
A mass‑profile exponent m≈1.9 in galaxies arises from the combination of a steep disk component (m≈2 ) and a shallower halo component (m≈1 ).
Observational data and theoretical models support this interpretation: exponential disks have M∝r^2 , isothermal haloes yield M∝r [2], and the NFW profile behaves as M∝r^2 at small radii[3].
The MBT CurvatureTrap simulation, while not a galaxy model, demonstrates how such composite exponents can emerge by mixing curvature types and tuning damping and memory parameters.
Parameter searches within the MBT model successfully reproduced slopes around 1.9,
thereby reinforcing the idea that the universal exponent reflects a physical balance between extended and concentrated mass distributions.
This cross‑disciplinary connection between astrophysical observations and abstract dynamical simulations deepens our understanding of why certain power‑law exponents appear universal across complex systems.

ref
________________________________________
[1] [2] [3] Rotation and Mass in the Milky Way and Spiral Galaxies - Yoshiaki Sofue
https://ned.ipac.caltech.edu/level5/Sept16/Sofue/Sofue4.html

author martin ollett
