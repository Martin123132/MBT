import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Load Pantheon+ data (same as before)
print("Fetching Pantheon+ data from GitHub...")
data_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
resp = requests.get(data_url)
if resp.status_code != 200:
    raise RuntimeError(f"Fetch failed: {resp.status_code}")
df = pd.read_csv(StringIO(resp.text), sep=r'\s+', comment='#',
                 dtype={'zCMB': float, 'MU_SH0ES': float, 'MU_SH0ES_ERR_DIAG': float})

z_pant = df['zCMB'].values
mu_obs = df['MU_SH0ES'].values
sig_mu = df['MU_SH0ES_ERR_DIAG'].values
m = (sig_mu > 0) & np.isfinite(mu_obs) & (z_pant > 0)
z_pant, mu_obs, sig_mu = z_pant[m], mu_obs[m], sig_mu[m]
print(f"Loaded {len(z_pant)} Pantheon+ SNe")

# BAO & CMB data
z_bao = np.array([0.38, 0.51, 0.61])
DM_obs = np.array([10.23, 13.67, 15.99])
sig_DM = np.array([0.12, 0.14, 0.19])
H_obs = np.array([80, 91, 100])
sig_H = np.array([2.0, 2.5, 3.0])

z_ls, l_obs, sig_l = 1090.0, 220.0, 1.0
c = 299792.458  # km/s

# ======================
# MODEL HIERARCHY - Start Simple, Add Complexity
# ======================

class MBTModelHierarchy:
    """Test different levels of model complexity"""
    
    def __init__(self):
        self.models = {
            'core_2param': self.core_geometric_model,
            'minimal_3param': self.minimal_model,
            'standard_4param': self.standard_model,
            'enhanced_6param': self.enhanced_model
        }
    
    # Model 1: Core 2-parameter geometric model
    def core_geometric_model(self, z, params):
        """Original geometric time idea: D(z) = (2c/H₀) × (1 - (1+z)^(-α×e^(β×z))) / (α×e^(β×z))"""
        alpha, H0 = params[:2]
        beta = -0.0005  # Fixed based on your earlier analysis
        
        # Your core geometric time dilation
        time_factor = alpha * np.exp(beta * z)
        term = (1 + z) ** (-time_factor)
        
        if np.any(time_factor <= 0) or np.any(term <= 0):
            return np.inf * np.ones_like(z)
            
        dM = (2 * c / H0) * (1 - term) / time_factor
        return dM * (1 + z)  # Convert to luminosity distance
    
    # Model 2: Add transition parameter
    def minimal_model(self, z, params):
        """Add one transition parameter for better high-z behavior"""
        alpha, H0, transition = params[:3]
        beta = -0.0005
        
        # Modified with transition
        base_factor = alpha * np.exp(beta * z)
        transition_factor = 1 + transition * z
        time_factor = base_factor * transition_factor
        
        term = (1 + z) ** (-time_factor)
        if np.any(time_factor <= 0) or np.any(term <= 0):
            return np.inf * np.ones_like(z)
            
        dM = (2 * c / H0) * (1 - term) / time_factor
        return dM * (1 + z)
    
    # Model 3: Standard model with log term
    def standard_model(self, z, params):
        """Add logarithmic term like your current model"""
        alpha, beta, H0, transition = params[:4]
        
        denom = 1 + alpha * np.log(1 + z) + beta * z
        if np.any(denom <= 0):
            return np.inf * np.ones_like(z)
            
        base = (c / H0) * (z * (1 + transition * z)) / denom
        return base * (1 + z)  # Convert to luminosity distance
    
    # Model 4: Enhanced model (your current 6-param version)
    def enhanced_model(self, z, params):
        """Full model with all terms"""
        alpha, beta, H0, transition, highz, quad = params[:6]
        
        denom = 1 + alpha * np.log(1 + z) + beta * z
        if np.any(denom <= 0):
            return np.inf * np.ones_like(z)
            
        base = (c / H0) * (z * (1 + transition * z)) / denom
        correction = (1 + highz * np.tanh(z / transition)) * (1 + quad * z * z)
        return base * correction

def test_model_hierarchy():
    """Test models in order of complexity"""
    
    mbt = MBTModelHierarchy()
    
    # Define parameter sets for each model
    model_configs = {
        'core_2param': {
            'init': [0.98, 70.0],
            'bounds': [(0.5, 1.5), (65, 80)],
            'param_names': ['α', 'H₀']
        },
        'minimal_3param': {
            'init': [0.98, 70.0, 0.02],
            'bounds': [(0.5, 1.5), (65, 80), (0.001, 0.1)],
            'param_names': ['α', 'H₀', 'transition']
        },
        'standard_4param': {
            'init': [0.20, 0.10, 70.0, 0.02],
            'bounds': [(0.05, 0.4), (0.01, 0.2), (65, 80), (0.001, 0.1)],
            'param_names': ['α', 'β', 'H₀', 'transition']
        },
        'enhanced_6param': {
            'init': [0.20, 0.10, 70.0, 0.02, 0.10, 0.00],
            'bounds': [(0.05, 0.4), (0.01, 0.2), (65, 80), (0.001, 0.1), (0.05, 0.5), (-0.05, 0.05)],
            'param_names': ['α', 'β', 'H₀', 'transition', 'highz', 'quad']
        }
    }
    
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n=== Testing {model_name} ===")
        
        def objective(params):
            # Calculate distances for all datasets
            try:
                # Pantheon distances
                dL_pant = mbt.models[model_name](z_pant, params)
                if np.any(np.isinf(dL_pant)) or np.any(dL_pant <= 0):
                    return 1e12
                mu_pred = 5 * np.log10(dL_pant) + 25
                
                # BAO distances (need to extract comoving distance)
                dL_bao = mbt.models[model_name](z_bao, params)
                dM_bao = dL_bao / (1 + z_bao)  # Convert back to comoving
                
                # Approximate H(z) from distance derivative
                dz = 0.001
                dL_plus = mbt.models[model_name](z_bao + dz, params)
                dL_minus = mbt.models[model_name](z_bao - dz, params)
                dM_plus = dL_plus / (1 + z_bao + dz)
                dM_minus = dL_minus / (1 + z_bao - dz)
                H_pred = c / ((dM_plus - dM_minus) / (2 * dz))
                
                # CMB distance
                dL_cmb = mbt.models[model_name](np.array([z_ls]), params)[0]
                dA_cmb = dL_cmb / (1 + z_ls)**2
                
                # Calculate chi-squared components
                chi2_sn = np.sum(((mu_obs - mu_pred) / sig_mu) ** 2)
                chi2_dm = np.sum(((DM_obs - dM_bao/147.0) / sig_DM) ** 2)  # Assume r_d ≈ 147
                chi2_h = np.sum(((H_obs - H_pred) / sig_H) ** 2)
                
                # CMB constraint (rough approximation)
                l1_pred = np.pi * dA_cmb / 150.0  # Assume r_s ≈ 150
                chi2_cmb = ((l_obs - l1_pred) / sig_l) ** 2
                
                # Normalized total
                total_chi2 = (chi2_sn/len(z_pant) + chi2_dm/len(z_bao) + 
                             chi2_h/len(z_bao) + chi2_cmb)
                
                return total_chi2
                
            except:
                return 1e12
        
        # Fit model
        try:
            res = minimize(objective, config['init'], bounds=config['bounds'], 
                          method='L-BFGS-B')
            
            results[model_name] = {
                'params': res.x,
                'chi2': res.fun,
                'success': res.success,
                'nparams': len(config['init']),
                'param_names': config['param_names']
            }
            
            print(f"Parameters: {dict(zip(config['param_names'], res.x))}")
            print(f"Normalized χ²: {res.fun:.4f}")
            print(f"AIC: {2*len(config['init']) + res.fun:.4f}")
            print(f"Success: {res.success}")
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            results[model_name] = None
    
    return results

def compare_models(results):
    """Compare model performance using information criteria"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    for model_name, result in valid_results.items():
        nparams = result['nparams']
        chi2 = result['chi2']
        aic = 2 * nparams + chi2
        bic = np.log(len(z_pant)) * nparams + chi2  # Rough approximation
        
        print(f"\n{model_name}:")
        print(f"  Parameters: {nparams}")
        print(f"  χ²/dof: {chi2:.4f}")
        print(f"  AIC: {aic:.4f}")
        print(f"  BIC: {bic:.4f}")
    
    # Find best model by AIC
    if valid_results:
        best_aic = min(valid_results.items(), 
                      key=lambda x: 2*x[1]['nparams'] + x[1]['chi2'])
        print(f"\nBest model by AIC: {best_aic[0]}")
        
        # Calculate AIC differences
        best_aic_value = 2*best_aic[1]['nparams'] + best_aic[1]['chi2']
        print(f"\nΔAIC relative to best model:")
        for model_name, result in valid_results.items():
            aic = 2*result['nparams'] + result['chi2']
            delta_aic = aic - best_aic_value
            print(f"  {model_name}: {delta_aic:.2f}")

# Run the analysis
if __name__ == "__main__":
    print("Testing MBT model hierarchy...")
    results = test_model_hierarchy()
    compare_models(results)
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("Choose the simplest model with ΔAIC < 2 from the best model")
    print("="*60)
