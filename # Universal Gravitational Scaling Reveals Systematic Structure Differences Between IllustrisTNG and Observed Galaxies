# Universal Gravitational Scaling Reveals Systematic Structure Differences Between IllustrisTNG and Observed Galaxies

**Martin Ollett**  
Independent Researcher  
October 2025

---

## Abstract

Recent analysis of gravitational systems spanning planetary orbits to galaxy clusters reveals that enclosed mass profiles follow simple power-law forms M(r) ∝ r^m across ten orders of magnitude in scale. For galaxies, analysis of 80 systems in the SPARC database yields a nearly universal exponent m = 1.878 ± 0.084. This steep slope indicates disk-dominated mass distributions where baryonic matter comprises 60-70% of the enclosed mass within optical radii. In contrast, analysis of 23 comparable galaxies from the IllustrisTNG simulation yields m = 0.895 ± 0.106, indicating significantly more concentrated, dark matter-dominated profiles. The 11.7σ discrepancy quantifies a systematic difference in galaxy structure between state-of-the-art simulations and observations, likely arising from insufficient core formation through baryonic feedback or limitations in the adopted dark matter model.

---

## 1. Introduction

### 1.1 The Universal Scaling Framework

Analysis of diverse gravitational systems reveals a simple relationship between enclosed mass M(r) and radius r. For systems in equilibrium where circular velocity v satisfies v²(r) = GM(r)/r, the enclosed mass often follows a power law M(r) ∝ r^m over substantial radial ranges. The exponent m characterizes the degree of mass concentration, with systematic values emerging across different astrophysical scales:

- **Planetary systems:** m ≈ 0.00 (point mass, Keplerian)
- **Binary stars:** m ≈ 0.00 (binary appears point-like from circumbinary orbits)
- **Galaxy clusters:** m ≈ 0.79 (diffuse dark matter + galaxies)
- **Individual galaxies:** m ≈ 1.88 (observed, SPARC)

This hierarchy spans over ten orders of magnitude in size and fifteen orders in mass. The power-law description appears to capture fundamental aspects of gravitational structure formation across diverse physical regimes.

### 1.2 Physical Interpretation of m

The exponent m directly connects to mass distribution profiles. An exponential stellar disk produces m ≈ 2 since enclosed mass scales as R² at small radii. A classic isothermal dark matter halo yields m ≈ 1 since enclosed mass grows linearly with radius. The Navarro-Frenk-White profile transitions between these behaviors. A composite system containing both disk and halo therefore produces an intermediate exponent between 1 and 2.

The observed value m ≈ 1.88 in SPARC galaxies requires that stellar and gaseous disks contribute the majority of enclosed mass within optical radii. A disk fraction of approximately 60-70% combined with only 30-40% from a more extended component reproduces the steep slope. This weighting is an empirical result: NFW-dominated profiles with m ≈ 1 cannot produce m ≈ 1.88. The universal exponent thus directly indicates baryonic dominance on kiloparsec scales.

### 1.3 Testing Simulations

Cosmological hydrodynamical simulations such as IllustrisTNG attempt to reproduce observed galaxy properties by incorporating gas cooling, star formation, supernova and AGN feedback within the ΛCDM framework. Despite sophisticated baryonic physics, tensions remain. The cusp-core problem describes the persistent difference between steep central density cusps predicted by dark matter-only simulations and the shallower cores preferred by observations of dwarf and low-surface-brightness galaxies.

Recent work using strong gravitational lensing has shown that even TNG's baryonic runs produce dark matter halos more concentrated than observations require. The m-exponent framework provides a complementary test: if TNG reproduces observed galaxy structure, simulated galaxies should yield m ≈ 1.88. Systematic deviations would indicate fundamental differences in mass distribution between simulations and reality.

---

## 2. Methodology

### 2.1 SPARC Sample Analysis

The SPARC database contains 175 nearby late-type galaxies with Spitzer 3.6 μm photometry and high-quality HI rotation curves. For this analysis, 80 galaxies were selected with:

- Stellar mass range: 10^9 - 10^11 M☉
- Quality rotation curves extending beyond 15 kpc
- Minimal non-circular motions
- Sufficient radial coverage for power-law fitting

For each galaxy:
1. Rotation velocities v(r) were extracted from published curves
2. Enclosed mass was calculated: M(r) = v²(r) × r / G
3. Power-law fits were performed in log-space over 5-25 kpc
4. Exponent m was extracted with quality threshold R² > 0.95

The resulting distribution shows remarkable convergence: mean m = 1.878, median m = 1.879, standard deviation σ = 0.084. The distribution is approximately Gaussian with 76% of galaxies within ±0.10 of the mean.

### 2.2 Mass Threshold and Systematic Trends

Analysis reveals a clear mass threshold at M_stellar ≈ 5×10^9 M☉:

- **Below threshold:** Higher scatter (σ = 0.078), diverse profiles
- **Above threshold:** Lower scatter (σ = 0.027), universal convergence to m ≈ 1.88

This threshold corresponds to circular velocities V_vir ≈ 50-70 km/s, suggesting a transition where efficient cooling and disk formation become established. Below this mass, supernova feedback can disrupt disk assembly, producing diverse exponents. Above it, stable disks form with consistent structure.

Morphological trends support the disk interpretation:
- Irregular galaxies: m = 1.880 ± 0.078
- Magellanic spirals: m = 1.854 ± 0.091  
- Early spirals (Sb): m = 1.872 ± 0.003 (remarkably uniform)
- Late spirals (Sd): m = 1.898 ± 0.068

Environmental processing also affects m. Cluster galaxies show m = 1.925 ± 0.003, approximately 0.067 higher than isolated systems (m = 1.858 ± 0.086). This shift likely reflects ram pressure stripping and tidal effects removing extended material.

### 2.3 IllustrisTNG Sample Analysis

The IllustrisTNG-100 simulation was analyzed at z=0 (snapshot 99). Selection criteria matched the SPARC sample:

- Stellar mass: 10^9 - 10^11 M☉
- Total particle count < 5×10^6 (memory constraint)
- Half-mass radius: 0.5-50 kpc (well-resolved systems)
- Quality fit requirement: R² > 0.95

For each galaxy:
1. Particle positions and masses were loaded for stars, gas, and dark matter
2. Radial distances from subhalo center were calculated
3. Cumulative enclosed mass profiles M(r) were constructed
4. Power-law fits were performed over 1-25 kpc
5. Exponent m was extracted

Dark matter particle masses were computed from total subhalo dark matter mass divided by particle count, as individual DM particle masses are not stored in snapshot files. The analysis used direct particle data rather than pre-computed profiles to ensure methodology consistency with SPARC analysis.

From 30 candidate galaxies, 23 satisfied quality criteria. Seven were excluded due to poor power-law fits (R² < 0.95), typically arising from recent mergers or disturbed morphologies.

---

## 3. Results

### 3.1 SPARC: Universal Convergence to m = 1.878

The 80-galaxy SPARC sample yields:

- Mean: m = 1.878
- Median: m = 1.879
- Standard deviation: σ = 0.084
- Range: [1.65, 2.10]

This distribution demonstrates universal convergence across diverse galaxy types in the 10^9 - 10^11 M☉ mass range. The narrow scatter indicates that galaxies reach a preferred mass distribution once sufficient mass accumulates for stable disk formation.

The composite interpretation requires:
- Stellar disk contribution: ~60-70% (m ≈ 2.0)
- Extended halo contribution: ~30-40% (m ≈ 1.0)
- Resulting composite: m ≈ 1.88

This weighting demonstrates baryonic dominance within optical radii. Dark matter-only profiles cannot reproduce the observed steep slope.

### 3.2 TNG: Systematic Concentration at m = 0.895

The 23 TNG galaxies yield:

- Mean: m = 0.895
- Median: m = 0.878
- Standard deviation: σ = 0.106
- Range: [0.71, 1.13]

No TNG galaxy approaches the SPARC distribution. Zero galaxies fall within ±0.20 of m = 1.878. The entire TNG distribution is shifted systematically toward lower values, indicating more concentrated mass profiles.

### 3.3 Quantifying the Discrepancy

**Direct comparison:**
- SPARC: m = 1.878 ± 0.084
- TNG: m = 0.895 ± 0.106
- Difference: Δm = 0.983
- Significance: 11.7σ

This represents approximately a factor-of-two difference in mass profile exponent. The discrepancy is not attributable to scatter, selection effects, or measurement uncertainties. It indicates a fundamental structural difference between simulated and observed galaxies.

### 3.4 Physical Implications

The exponent m connects directly to rotation curve shape through v(r) ∝ r^((m-1)/2):

**SPARC (m ≈ 1.88):**
- v(r) ∝ r^0.44
- Rising rotation curves
- Baryonic disk dominates inner regions
- Dark matter contribution increases outward

**TNG (m ≈ 0.90):**
- v(r) ∝ r^-0.05 ≈ constant
- Flat to slightly declining rotation curves
- Dark matter dominates at all radii
- Weak disk influence on gravitational potential

The low m in TNG indicates that dark matter halos remain concentrated despite baryonic physics. Stellar and AGN feedback have not created the extended mass distributions observed in real galaxies.

---

## 4. Discussion

### 4.1 The Cusp-Core Problem Quantified

The systematic difference between TNG and SPARC provides a new quantification of the long-standing cusp-core problem. ΛCDM simulations generically produce cuspy density profiles (ρ ∝ r^-1) in galaxy centers, corresponding to relatively flat M(r) profiles with m ≲ 1.2. Observations prefer extended profiles with m approaching 2.0, indicating either cored dark matter distributions or strong baryonic dominance.

The m-exponent diagnostic reveals that IllustrisTNG, despite incorporating sophisticated baryonic physics, still produces galaxies approximately twice as concentrated as observations require. This suggests:

1. **Insufficient feedback:** Stellar and AGN feedback may not redistribute matter efficiently enough to flatten inner profiles
2. **Adiabatic contraction:** Baryonic infall could deepen potential wells, pulling dark matter inward
3. **Physics beyond ΛCDM:** Self-interacting dark matter or modified gravity might be required
4. **Resolution limitations:** Higher resolution might increase m values, though the complete absence of m > 1.1 suggests a robust trend

### 4.2 Comparison with Previous Work

Extensive literature documents discrepancies between simulated and observed rotation curves. The m-exponent framework provides a complementary single-parameter diagnostic that makes no assumptions about halo profile shapes or decomposition methods.

Previous work using strong gravitational lensing has shown that TNG halos are more concentrated than observations require. The current analysis confirms this using rotation curve data and direct mass profiles, demonstrating consistency across independent probes.

The Baryonic Tully-Fisher relation shows tight correlations between baryonic mass and rotation velocity. Galaxies with higher m (more extended) may exhibit different BTF behavior than concentrated systems, potentially explaining residual BTF scatter.

### 4.3 Possible Physical Origins

Several mechanisms could contribute to TNG's low m values:

**Adiabatic contraction:** Gas cooling and star formation deepen gravitational wells. Dark matter responds by contracting inward, steepening inner density profiles. The AIDA-TNG simulations demonstrate that baryonic physics increases halo concentrations relative to dark matter-only runs.

**Weak feedback:** Supernova and AGN feedback can drive outflows that redistribute dark matter and create cores. TNG implements both processes, but the analysis suggests insufficient strength or burstiness. More efficient feedback could flatten profiles and increase m.

**Gas fraction effects:** SPARC shows anti-correlation between gas fraction and m. Gas-rich galaxies (f_gas > 0.6) yield m ≈ 1.79, while gas-poor systems (f_gas < 0.3) reach m ≈ 1.89. TNG galaxies may undergo earlier or more efficient star formation, reducing gas fractions and concentrating mass. However, this effect alone cannot produce m ≈ 0.90.

**Dark matter model:** Alternative dark matter physics could reduce central densities. Self-interacting dark matter produces lower halo concentrations in AIDA-TNG simulations. Warm dark matter suppresses small-scale structure formation, potentially affecting inner profiles. Modified gravity theories eliminate dark matter entirely, with baryons alone determining structure.

### 4.4 Environmental and Mass Trends

Both SPARC and TNG show environmental effects on m. Cluster galaxies in SPARC have higher m than field galaxies, likely due to ram pressure stripping of extended material. TNG should reproduce this trend if environmental processes are implemented correctly.

The mass threshold at 5×10^9 M☉ in SPARC marks the transition to universal m ≈ 1.88. TNG galaxies show no such threshold within the analyzed range, suggesting they never reach the disk-dominated regime characteristic of real galaxies.

### 4.5 Implications for Galaxy Formation Theory

The m ≈ 1.88 value in SPARC indicates efficient angular momentum conservation and disk formation. The threshold mass corresponds to the scale where gas cooling becomes efficient and disks stabilize against feedback disruption.

TNG's failure to reproduce this structure suggests issues with:
- **Feedback timing:** Feedback may occur too early or too late to shape halos effectively
- **Feedback geometry:** Isotropic feedback might be less efficient than collimated outflows
- **Halo response:** Dark matter may not respond appropriately to baryon redistribution
- **Initial conditions:** Cosmological initial conditions or halo assembly histories might differ systematically

### 4.6 Future Directions

Several avenues could resolve the discrepancy:

1. **Higher resolution:** TNG-50 provides factor-of-eight better mass resolution. Analysis could determine whether m increases with resolution.

2. **Alternative simulations:** FIRE-2, EAGLE, and NewHorizon simulations use different feedback prescriptions. Comparative analysis would identify whether low m is universal or TNG-specific.

3. **Modified feedback:** Adjusting feedback parameters in post-processing or new runs could test which modifications increase m toward observed values.

4. **Alternative dark matter:** AIDA-TNG includes self-interacting dark matter runs. Analysis of these could quantify how SIDM affects m.

5. **Redshift evolution:** Tracking individual galaxies backward in time could reveal when and why m diverges from observations.

---

## 5. Conclusions

1. **Universal diagnostic validated:** The m-exponent provides a simple, model-independent measure of mass concentration applicable from planetary systems to galaxy clusters. The framework spans over ten orders of magnitude in scale.

2. **SPARC universality confirmed:** Real galaxies in the mass range 10^9 - 10^11 M☉ converge to m = 1.878 ± 0.084. Above a threshold at 5×10^9 M☉, scatter drops dramatically (σ = 0.027), indicating a universal structure emerges once stable disks form.

3. **TNG systematic offset quantified:** IllustrisTNG galaxies yield m = 0.895 ± 0.106, representing an 11.7σ discrepancy from observations. No simulated galaxy approaches the observed distribution.

4. **Physical interpretation:** The discrepancy indicates that TNG galaxies are approximately twice as centrally concentrated as real galaxies. TNG produces dark matter-dominated systems with flat rotation curves, while SPARC shows disk-dominated systems with rising rotation curves.

5. **Likely causes:** The offset probably arises from insufficient core formation through baryonic feedback, adiabatic contraction of dark matter halos, or fundamental limitations in the adopted dark matter model.

6. **Predictive framework:** The m-exponent enables quantitative benchmarking of future simulations. Success requires increasing m from ~0.90 to ~1.88, providing a clear target for improved physics implementations.

The simple exponent m thus serves as a sensitive diagnostic of galaxy formation physics. The systematic difference between IllustrisTNG and SPARC observations highlights remaining challenges in reproducing observed galaxy structure within the standard ΛCDM framework.

---

## Acknowledgments

The author thanks Dylan Nelson for providing access to the IllustrisTNG JupyterLab environment. This work made use of the IllustrisTNG simulations and the SPARC database. Analysis was performed using NumPy, SciPy, and Matplotlib.

---

## References

Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157  
Nelson, D., et al. 2019, Computational Astrophysics and Cosmology, 6, 2  
Pillepich, A., et al. 2018, MNRAS, 473, 4077  

[Additional references to cusp-core problem, AIDA-TNG, alternative dark matter models, etc.]

---

## Data Availability

Analysis code: https://github.com/Martin123132/mts-galaxy-analysis  
SPARC data: http://astroweb.cwru.edu/SPARC/  
IllustrisTNG data: https://www.tng-project.org/data/
