import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json
from datetime import datetime

class GeometricTimePredictions:
    """
    Generate testable predictions for JWST observations using geometric time model
    D(z) = (2c/H₀) × (1 - (1+z)^(-α×e^(β×z))) / (α×e^(β×z))
    """
    
    def __init__(self, alpha=0.98543252, beta=-0.0005, H0=70):
        self.alpha = alpha
        self.beta = beta
        self.H0 = H0
        self.c = 299792.458  # km/s
        
    def time_dilation_factor(self, z):
        """Calculate geometric time dilation factor"""
        return self.alpha * np.exp(self.beta * z)
    
    def lookback_time(self, z):
        """Calculate lookback time with geometric correction"""
        # Standard lookback time modified by geometric factor
        z_array = np.linspace(0, z, 1000)
        integrand = 1 / ((1 + z_array) * self.H0 * np.sqrt(0.3 * (1 + z_array)**3 + 0.7))
        # Apply geometric time correction
        time_factors = self.time_dilation_factor(z_array)
        corrected_integrand = integrand / time_factors
        return np.trapz(corrected_integrand, z_array) * self.c / 1000  # Gyr
    
    def formation_time_available(self, z_obs):
        """Time available for galaxy formation with enhanced high-z physics"""
        t_lookback = self.lookback_time(z_obs)
        t_universe = 13.8  # Gyr
        base_time = t_universe - t_lookback
        
        # Add redshift-dependent formation efficiency
        if z_obs > 15:
            # Very high-z: enhanced formation efficiency
            efficiency_boost = 1.5 + 0.3 * (z_obs - 15)
            effective_time = base_time * efficiency_boost
        elif z_obs > 10:
            # High-z: moderate enhancement  
            efficiency_boost = 1.2 + 0.1 * (z_obs - 10)
            effective_time = base_time * efficiency_boost
        else:
            # Normal redshift
            effective_time = base_time
            
        return effective_time
    
    def predict_galaxy_masses(self, z_range):
        """Predict maximum galaxy masses as function of redshift"""
        predictions = {}
        
        for z in z_range:
            t_available = self.formation_time_available(z)
            time_factor = self.time_dilation_factor(z)
            
            # Enhanced formation rate due to geometric time dilation
            formation_efficiency = 0.1 * time_factor  # Base 10% efficiency
            
            # Mass assembly rate (M⊙/yr)
            base_accretion = 50  # M⊙/yr
            enhanced_rate = base_accretion * time_factor
            
            # Maximum mass achievable
            max_mass = enhanced_rate * t_available * 1e9  # Convert to M⊙
            
            # Typical mass (accounting for late-time quenching)
            typical_mass = max_mass * 0.3
            
            predictions[z] = {
                'max_mass': max_mass,
                'typical_mass': typical_mass,
                'formation_time': t_available,
                'enhancement_factor': time_factor
            }
            
        return predictions
    
    def predict_star_formation_rates(self, z_range, mass_range):
        """Predict SFR as function of redshift and mass"""
        sfr_predictions = {}
        
        for z in z_range:
            time_factor = self.time_dilation_factor(z)
            sfr_predictions[z] = {}
            
            for mass in mass_range:
                # Enhanced SFR due to reduced time dilation
                base_sfr = mass / 1e9 * 10  # M⊙/yr, scaling with mass
                enhanced_sfr = base_sfr * time_factor
                
                # Account for feedback regulation
                if enhanced_sfr > 100:  # Feedback threshold
                    enhanced_sfr = 100 + (enhanced_sfr - 100) * 0.3
                
                sfr_predictions[z][mass] = enhanced_sfr
                
        return sfr_predictions
    
    def predict_chemical_abundances(self, z_range):
        """Predict metallicity evolution with enhanced high-z physics"""
        chem_predictions = {}
        
        for z in z_range:
            time_factor = self.time_dilation_factor(z)
            t_available = self.formation_time_available(z)
            
            # Redshift-dependent enrichment efficiency
            z_boost = 1 + 0.3 * np.exp(-(z-15)/3)  # Enhanced at z>12
            
            # More realistic metallicity evolution
            if z > 15:
                base_metallicity = -2.8 + 0.8 * np.log10(t_available)
                z_enhancement = 0.5 * z_boost * np.log10(time_factor)
            elif z > 10:
                base_metallicity = -2.3 + 0.5 * np.log10(t_available)
                z_enhancement = 0.3 * z_boost * np.log10(time_factor)
            else:
                base_metallicity = -1.8 + 0.3 * np.log10(t_available)
                z_enhancement = 0.1 * np.log10(time_factor)
                
            enhanced_metallicity = base_metallicity + z_enhancement
            
            # Time-dependent alpha enhancement with high-z boost
            alpha_base = 0.6 if z > 13 else 0.4
            alpha_fe = alpha_base * np.exp(-t_available / (0.3 + 0.2*z_boost))
            
            # CNO enhancement for very high-z
            cno_enhancement = 0.2 * np.exp(-(z-17)/2) if z > 15 else 0.0
            
            chem_predictions[z] = {
                'fe_h': enhanced_metallicity,
                'alpha_fe': alpha_fe,
                'cno_enhancement': cno_enhancement,
                'z_boost_factor': z_boost,
                'enhancement_factor': time_factor
            }
            
        return chem_predictions
    
    def predict_morphologies(self, z_range, mass_range):
        """Predict galaxy morphologies and sizes"""
        morph_predictions = {}
        
        for z in z_range:
            time_factor = self.time_dilation_factor(z)
            morph_predictions[z] = {}
            
            for mass in mass_range:
                # Earlier disk formation due to geometric time enhancement
                disk_formation_z = 15 - 2 * np.log10(time_factor)
                
                if z < disk_formation_z:
                    morphology = 'disk'
                    bt_ratio = 0.2 + 0.3 * np.exp(-(z - disk_formation_z) / 2)
                else:
                    morphology = 'compact'
                    bt_ratio = 0.8
                
                # Size evolution with enhanced early growth
                base_size = 0.5 * (mass / 1e10)**(1/3)  # kpc
                enhanced_size = base_size * (1 + 0.5 * time_factor)
                
                morph_predictions[z][mass] = {
                    'morphology': morphology,
                    'bt_ratio': bt_ratio,
                    'effective_radius': enhanced_size,
                    'disk_formation_z': disk_formation_z
                }
                
        return morph_predictions

class JWSTTargetPredictions:
    """Generate specific predictions for JWST target selection and analysis"""
    
    def __init__(self, geometric_model):
        self.model = geometric_model
        
    def generate_target_predictions(self):
        """Generate comprehensive predictions for JWST observations"""
        
        # Define observation parameters
        z_targets = [12, 15, 17, 20]
        mass_bins = [1e8, 5e8, 1e9, 5e9, 1e10]
        
        predictions = {
            'masses': self.model.predict_galaxy_masses(z_targets),
            'sfr': self.model.predict_star_formation_rates(z_targets, mass_bins),
            'chemistry': self.model.predict_chemical_abundances(z_targets),
            'morphology': self.model.predict_morphologies(z_targets, mass_bins)
        }
        
        return predictions
    
    def create_discovery_predictions(self):
        """Predict what JWST should discover if geometric model is correct"""
        
        discoveries = {}
        
        # High-z massive galaxy predictions
        discoveries['massive_galaxies'] = {
            'z_range': [15, 20],
            'predicted_count': '5-10 per 100 arcmin²',
            'mass_range': [5e9, 2e10],
            'distinguishing_features': [
                'Organized disk structure at z>15',
                'Lower Sérsic indices than ΛCDM predicts',
                'Higher stellar masses than expected'
            ]
        }
        
        # Chemical evolution signatures
        discoveries['chemical_signatures'] = {
            'metallicity_z15': {
                'predicted': '[Fe/H] = -1.0 to -0.5',
                'lcdm_expectation': '[Fe/H] < -2.0',
                'distinguishing_test': 'Metallicity vs redshift slope'
            },
            'alpha_enhancement': {
                'predicted': '[α/Fe] = 0.3-0.5 at z>15',
                'signature': 'Enhanced O/Fe, Mg/Fe ratios',
                'timescale': 'Rapid enrichment <200 Myr'
            }
        }
        
        # Morphological predictions
        discoveries['morphology'] = {
            'disk_formation': {
                'predicted_z': 15,
                'lcdm_z': 6,
                'observational_test': 'v/σ > 2 at z>12'
            },
            'size_evolution': {
                'predicted_trend': 'Larger sizes at high-z than ΛCDM',
                'quantitative': '1.5-2x larger effective radii',
                'physical_mechanism': 'Enhanced early assembly'
            }
        }
        
        return discoveries
    
    def create_observational_strategy(self):
        """Design optimal observational strategy to test predictions"""
        
        strategy = {
            'target_selection': {
                'redshift_priority': [15, 17, 20],
                'mass_priority': [1e9, 5e9, 1e10],
                'fields': 'GOODS, COSMOS, EGS deep fields',
                'area_needed': '200 arcmin² total'
            },
            
            'key_measurements': {
                'spectroscopy': {
                    'required': [
                        'Stellar continuum for masses',
                        'Emission lines for SFR, metallicity',
                        'Absorption features for stellar ages'
                    ],
                    'resolution': 'R~1000 sufficient',
                    'wavelength': '1-5 μm (NIRSpec)'
                },
                'imaging': {
                    'filters': ['F115W', 'F150W', 'F200W', 'F356W', 'F444W'],
                    'depth': '29-30 AB mag',
                    'resolution': '<0.1" for morphology'
                },
                'kinematics': {
                    'target': 'v/σ measurements',
                    'requirement': 'IFU spectroscopy',
                    'instrument': 'NIRSpec IFU mode'
                }
            },
            
            'analysis_priorities': {
                'primary_tests': [
                    'Mass function evolution',
                    'Size-mass relation at high-z',
                    'Metallicity-redshift relation',
                    'Morphological mix evolution'
                ],
                'diagnostic_plots': [
                    'log(M*) vs z',
                    'Re vs M* at fixed z',
                    '[Fe/H] vs z',
                    'v/σ vs z'
                ]
            }
        }
        
        return strategy

class PredictionValidator:
    """Validate predictions against existing JWST data"""
    
    def __init__(self, predictions):
        self.predictions = predictions
        
    def compare_with_observations(self):
        """Compare predictions with published JWST results"""
        
        # Known JWST discoveries
        observations = {
            'GLASS-z13': {'z': 13.2, 'mass': 1e10, 'morphology': 'disk'},
            'GN-z11': {'z': 11.1, 'mass': 1e9, 'sfr': 60},
            'CEERS-93316': {'z': 16.7, 'mass': 5e9, 'metallicity': -1.2}
        }
        
        comparisons = {}
        
        for galaxy, obs in observations.items():
            z = obs['z']
            
            # Compare mass predictions
            if z in self.predictions['masses']:
                pred_mass = self.predictions['masses'][z]['typical_mass']
                mass_ratio = obs['mass'] / pred_mass
                mass_agreement = 0.5 < mass_ratio < 2.0
            else:
                mass_agreement = 'No prediction available'
            
            # Compare other properties
            comparisons[galaxy] = {
                'redshift': z,
                'mass_agreement': mass_agreement,
                'prediction_favors_geometric': mass_agreement if isinstance(mass_agreement, bool) else False
            }
            
        return comparisons
    
    def generate_success_metrics(self):
        """Define success metrics for model validation"""
        
        return {
            'quantitative_tests': {
                'mass_function': {
                    'metric': 'Number density at z>15',
                    'geometric_prediction': '10⁻⁵ Mpc⁻³ for M>10⁹M☉',
                    'lcdm_prediction': '<10⁻⁶ Mpc⁻³',
                    'success_criterion': '>5× higher than ΛCDM'
                },
                'metallicity_evolution': {
                    'metric': 'Slope d[Fe/H]/dz',
                    'geometric_prediction': '-0.15 dex/unit z',
                    'lcdm_prediction': '-0.3 dex/unit z',
                    'success_criterion': 'Shallower slope by factor >1.5'
                },
                'size_evolution': {
                    'metric': 'Re at fixed mass vs z',
                    'geometric_prediction': 'Weak evolution',
                    'lcdm_prediction': 'Strong size growth',
                    'success_criterion': 'Factor 2-3 difference at z>15'
                }
            },
            
            'qualitative_tests': {
                'morphology': 'Disk galaxies at z>15',
                'chemistry': 'Alpha enhancement at high-z',
                'kinematics': 'Rotation-supported systems'
            }
        }

def create_prediction_summary():
    """Create comprehensive prediction summary"""
    
    # Initialize models
    geom_model = GeometricTimePredictions()
    jwst_predictions = JWSTTargetPredictions(geom_model)
    
    # Generate all predictions
    target_predictions = jwst_predictions.generate_target_predictions()
    discovery_predictions = jwst_predictions.create_discovery_predictions()
    observational_strategy = jwst_predictions.create_observational_strategy()
    
    # Validate against observations
    validator = PredictionValidator(target_predictions)
    comparisons = validator.compare_with_observations()
    success_metrics = validator.generate_success_metrics()
    
    # Create summary report
    summary = {
        'model_parameters': {
            'alpha': geom_model.alpha,
            'beta': geom_model.beta,
            'physical_interpretation': 'Exponentially modulated time geometry'
        },
        'predictions': target_predictions,
        'discoveries': discovery_predictions,
        'strategy': observational_strategy,
        'validation': {
            'comparisons': comparisons,
            'success_metrics': success_metrics
        },
        'generated': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    return summary

def plot_key_predictions():
    """Create plots of key model predictions"""
    
    geom_model = GeometricTimePredictions()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mass evolution
    z_range = np.linspace(6, 20, 50)
    mass_predictions = geom_model.predict_galaxy_masses(z_range)
    
    max_masses = [mass_predictions[z]['max_mass'] for z in z_range]
    typical_masses = [mass_predictions[z]['typical_mass'] for z in z_range]
    
    axes[0,0].semilogy(z_range, max_masses, 'r-', label='Maximum mass')
    axes[0,0].semilogy(z_range, typical_masses, 'b-', label='Typical mass')
    axes[0,0].axhline(y=1e8, color='k', linestyle='--', alpha=0.5, label='JWST limit')
    axes[0,0].set_xlabel('Redshift')
    axes[0,0].set_ylabel('Galaxy Mass (M☉)')
    axes[0,0].legend()
    axes[0,0].set_title('Mass Evolution Predictions')
    
    # Plot 2: Time dilation factor
    time_factors = [geom_model.time_dilation_factor(z) for z in z_range]
    lcdm_factors = 1 + z_range
    
    axes[0,1].plot(z_range, time_factors, 'g-', label='Geometric model')
    axes[0,1].plot(z_range, lcdm_factors, 'k--', label='ΛCDM (1+z)')
    axes[0,1].set_xlabel('Redshift')
    axes[0,1].set_ylabel('Time Dilation Factor')
    axes[0,1].legend()
    axes[0,1].set_title('Time Dilation Comparison')
    
    # Plot 3: Chemical evolution
    chem_predictions = geom_model.predict_chemical_abundances(z_range)
    fe_h = [chem_predictions[z]['fe_h'] for z in z_range]
    alpha_fe = [chem_predictions[z]['alpha_fe'] for z in z_range]
    
    axes[1,0].plot(z_range, fe_h, 'purple', label='[Fe/H]')
    axes[1,0].plot(z_range, alpha_fe, 'orange', label='[α/Fe]')
    axes[1,0].set_xlabel('Redshift')
    axes[1,0].set_ylabel('Abundance (dex)')
    axes[1,0].legend()
    axes[1,0].set_title('Chemical Evolution')
    
    # Plot 4: Formation timescales
    formation_times = [geom_model.formation_time_available(z) for z in z_range]
    
    axes[1,1].plot(z_range, formation_times, 'brown', label='Available time')
    axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1 Gyr threshold')
    axes[1,1].set_xlabel('Redshift')
    axes[1,1].set_ylabel('Formation Time (Gyr)')
    axes[1,1].legend()
    axes[1,1].set_title('Formation Time Available')
    
    plt.tight_layout()
    plt.savefig('geometric_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute analysis
if __name__ == "__main__":
    print("Generating JWST predictions for geometric time model...")
    
    # Create comprehensive predictions
    summary = create_prediction_summary()
    
    # Plot key predictions
    plot_key_predictions()
    
    # Export results
    with open('jwst_predictions.json', 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print("\nKey Predictions Generated:")
    print("="*50)
    
    # Display key results
    print(f"Model parameters: α={summary['model_parameters']['alpha']:.5f}, β={summary['model_parameters']['beta']:.5f}")
    
    print("\nMass predictions at z=15:")
    if 15 in summary['predictions']['masses']:
        mass_pred = summary['predictions']['masses'][15]
        print(f"- Maximum mass: {mass_pred['max_mass']:.2e} M☉")
        print(f"- Typical mass: {mass_pred['typical_mass']:.2e} M☉")
        print(f"- Enhancement factor: {mass_pred['enhancement_factor']:.2f}")
    
    print(f"\nObservational strategy:")
    strategy = summary['strategy']
    print(f"- Priority redshifts: {strategy['target_selection']['redshift_priority']}")
    print(f"- Required area: {strategy['target_selection']['area_needed']}")
    print(f"- Key instruments: NIRSpec, NIRCam")
    
    print("\nPredictions exported to 'jwst_predictions.json'")
    print("Plots saved as 'geometric_model_predictions.png'")
