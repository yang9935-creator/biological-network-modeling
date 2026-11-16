"""
Glucose Metabolic Network: From Multi-Compartment Model to AI-Discovered Parsimonious Equation
==============================================================================================
This code implements:
1. A complete 6-compartment differential equation system (as the "real world")
2. Generation of blood glucose dynamics data under different scenarios
3. Symbolic regression to discover parsimonious nonlinear equations from data
4. Visualization comparing the full system vs. the parsimonious equation
5. Saves all results to timestamped folders
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# Helper function to create results folders
# ============================================================================

def create_results_folder():
    """Create timestamped and latest results under the repository results directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_root = os.path.join(project_root, 'results', 'glucose_regulation')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(output_root, 'runs', timestamp)
    os.makedirs(results_folder, exist_ok=True)

    latest_folder = output_root
    os.makedirs(latest_folder, exist_ok=True)

    return results_folder, latest_folder

# ============================================================================
# Part 1: Complete Glucose Metabolic Network Model (6 compartments)
# ============================================================================

class GlucoseMetabolismModel:
    """
    Complete glucose metabolic network model
    Includes: Gut, Blood, Liver, Muscle, Adipose, Brain
    """
    
    def __init__(self):
        # Volume parameters (L)
        self.V_blood = 5.0      # Blood volume
        self.V_liver = 1.5      # Liver volume
        self.V_muscle = 25.0    # Muscle volume
        self.V_fat = 15.0       # Adipose tissue volume
        self.V_brain = 1.2      # Brain volume
        
        # Transport parameters
        self.k_abs = 0.05        # Gut absorption rate
        self.k_renal = 0.002     # Renal excretion rate (activated only when glucose > 10)
        
        # Michaelis-Menten constants
        self.Km_brain = 1.0      # Low Km for brain (high affinity)
        self.Km_muscle = 5.0     # Km for muscle uptake
        self.Km_fat = 5.0        # Km for adipose uptake
        self.Km_liver_uptake = 8.0  # Km for liver uptake
        
        # Maximum rate parameters
        self.Vmax_brain = 1.2     # Maximum brain uptake rate (mmol/min)
        self.Vmax_muscle_base = 3.0  # Baseline maximum muscle uptake
        self.Vmax_fat_base = 1.0     # Baseline maximum adipose uptake
        self.Vmax_liver_uptake = 2.0 # Maximum liver uptake
        
        # Hormonal regulation parameters
        self.insulin_sensitivity = 0.5  # Insulin sensitivity
        self.basal_insulin = 5.0        # Basal insulin level (mU/L)
        self.glucagon_sensitivity = 0.3 # Glucagon sensitivity
        
        # Liver production parameters
        self.liver_production_max = 2.0  # Maximum hepatic glucose output
        self.glycogen_storage = 100.0    # Initial glycogen储备
        
        # Consumption parameters
        self.muscle_consumption_base = 0.05  # Baseline muscle consumption
        self.brain_consumption = 0.8         # Constant brain consumption
        self.fat_synthesis_rate = 0.02       # Fat synthesis rate
        
    def insulin_dynamics(self, G_blood):
        """Insulin secretion model: glucose-dependent S-curve"""
        # Hill function: high glucose stimulates insulin secretion
        G0 = 5.0  # Glucose threshold
        n = 3     # Hill coefficient
        max_insulin = 50.0
        
        insulin = self.basal_insulin + (max_insulin - self.basal_insulin) * (G_blood**n) / (G0**n + G_blood**n)
        return insulin
    
    def glucagon_dynamics(self, G_blood):
        """Glucagon secretion model: low glucose stimulates secretion"""
        G0 = 4.0
        n = 2
        max_glucagon = 20.0
        
        # Low glucose stimulates glucagon secretion
        glucagon = max_glucagon * (G0**n) / (G0**n + G_blood**n)
        return glucagon
    
    def muscle_uptake(self, G_blood, insulin):
        """Muscle uptake: insulin-dependent GLUT4 transport"""
        # Insulin-stimulated uptake enhancement
        insulin_effect = 1.0 + self.insulin_sensitivity * (insulin - self.basal_insulin)
        insulin_effect = max(1.0, min(5.0, insulin_effect))  # Limit range
        
        Vmax = self.Vmax_muscle_base * insulin_effect
        
        # Michaelis-Menten kinetics
        uptake = Vmax * G_blood / (self.Km_muscle + G_blood)
        return uptake
    
    def fat_uptake(self, G_blood, insulin):
        """Adipose uptake: insulin-dependent"""
        insulin_effect = 1.0 + self.insulin_sensitivity * (insulin - self.basal_insulin)
        insulin_effect = max(1.0, min(5.0, insulin_effect))
        
        Vmax = self.Vmax_fat_base * insulin_effect
        uptake = Vmax * G_blood / (self.Km_fat + G_blood)
        return uptake
    
    def brain_uptake(self, G_blood):
        """Brain uptake: insulin-independent, high affinity"""
        # Can uptake even at low glucose
        uptake = self.Vmax_brain * G_blood / (self.Km_brain + G_blood)
        # Brain consumption is relatively constant
        consumption = min(self.brain_consumption, uptake)
        return consumption
    
    def liver_uptake(self, G_blood, insulin):
        """Liver uptake: storage as glycogen"""
        # High insulin promotes liver uptake
        insulin_effect = 1.0 + 0.3 * (insulin - self.basal_insulin)
        insulin_effect = max(1.0, min(3.0, insulin_effect))
        
        Vmax = self.Vmax_liver_uptake * insulin_effect
        uptake = Vmax * G_blood / (self.Km_liver_uptake + G_blood)
        return uptake
    
    def liver_production(self, G_blood, insulin, glucagon, glycogen):
        """Liver glucose production: glycogenolysis and gluconeogenesis"""
        # Insulin inhibits production, glucagon stimulates production
        insulin_inhibition = 1.0 / (1.0 + 0.1 * (insulin - self.basal_insulin))
        insulin_inhibition = max(0.2, min(1.0, insulin_inhibition))
        
        glucagon_stimulation = 1.0 + self.glucagon_sensitivity * glucagon
        glucagon_stimulation = min(3.0, glucagon_stimulation)
        
        # Glycogen availability effect
        glycogen_factor = min(1.0, glycogen / 50.0)
        
        production = (self.liver_production_max * insulin_inhibition * 
                     glucagon_stimulation * glycogen_factor)
        
        return production
    
    def renal_excretion(self, G_blood):
        """Renal excretion: occurs only above renal threshold"""
        threshold = 10.0  # Renal glucose threshold (mmol/L)
        if G_blood > threshold:
            return self.k_renal * (G_blood - threshold)
        else:
            return 0.0
    
    def model_equations(self, state, t, meal_input=0):
        """
        Complete system of differential equations
        state: [G_lumen, G_blood, G_liver, G_muscle, G_fat, G_brain, glycogen]
        """
        G_lumen, G_blood, G_liver, G_muscle, G_fat, G_brain, glycogen = state
        
        # Calculate hormone levels
        insulin = self.insulin_dynamics(G_blood)
        glucagon = self.glucagon_dynamics(G_blood)
        
        # Flux calculations
        J_abs = self.k_abs * G_lumen if G_lumen > 0 else 0
        
        J_muscle_up = self.muscle_uptake(G_blood, insulin)
        J_fat_up = self.fat_uptake(G_blood, insulin)
        J_brain_up = self.brain_uptake(G_blood)
        J_liver_up = self.liver_uptake(G_blood, insulin)
        J_liver_prod = self.liver_production(G_blood, insulin, glucagon, glycogen)
        J_renal = self.renal_excretion(G_blood)
        
        # Consumption
        J_muscle_cons = self.muscle_consumption_base * G_muscle
        J_fat_storage = self.fat_synthesis_rate * G_fat
        
        # Differential equations
        dG_lumen = -J_abs + meal_input
        
        dG_blood = (J_abs + J_liver_prod - J_muscle_up - J_fat_up - 
                   J_brain_up - J_liver_up - J_renal) / self.V_blood
        
        dG_liver = (J_liver_up - J_liver_prod) / self.V_liver
        
        dG_muscle = (J_muscle_up - J_muscle_cons) / self.V_muscle
        
        dG_fat = (J_fat_up - J_fat_storage) / self.V_fat
        
        dG_brain = (J_brain_up - self.brain_consumption) / self.V_brain
        
        # Liver glycogen dynamics
        dglycogen = J_liver_up - J_liver_prod - 0.001 * glycogen
        
        return [dG_lumen, dG_blood, dG_liver, dG_muscle, dG_fat, dG_brain, dglycogen]


# ============================================================================
# Part 2: Generate Training Data
# ============================================================================

def generate_training_data(model, scenarios):
    """
    Generate blood glucose dynamics data for different scenarios
    """
    all_t = []
    all_G = []
    all_labels = []
    all_raw_data = []  # Store raw data for saving
    
    # Initial state
    initial_state = [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0]  # Normal fasting state
    
    for scenario_name, meal_times, meal_amounts in scenarios:
        print(f"Generating scenario: {scenario_name}")
        
        # Time points (0-480 minutes, 8 hours)
        t = np.linspace(0, 480, 100)
        
        # Simulate
        def meal_function(t_val):
            """Meal input function"""
            meal = 0
            for meal_time, amount in zip(meal_times, meal_amounts):
                if meal_time <= t_val < meal_time + 30:  # Lasts 30 minutes
                    meal += amount / 30  # Average rate
            return meal
        
        # Solve ODE
        def ode_wrapper(state, t_val):
            return model.model_equations(state, t_val, meal_function(t_val))
        
        solution = odeint(ode_wrapper, initial_state, t)
        G_blood = solution[:, 1]  # Extract blood glucose
        
        all_t.append(t)
        all_G.append(G_blood)
        all_labels.append(scenario_name)
        all_raw_data.append({
            'time': t.tolist(),
            'glucose': G_blood.tolist(),
            'scenario': scenario_name,
            'meal_times': meal_times,
            'meal_amounts': meal_amounts
        })
    
    return all_t, all_G, all_labels, all_raw_data


# ============================================================================
# Part 3: Symbolic Regression - AI Discovering Parsimonious Equations
# ============================================================================

class SymbolicRegressor:
    """
    Simplified symbolic regression implementation
    Uses genetic programming concepts to find the best nonlinear differential equation
    """
    
    def __init__(self, population_size=50, generations=20):
        self.population_size = population_size
        self.generations = generations
        self.best_equation = None
        self.best_fitness = float('inf')
        self.template_idx = None
        
    @staticmethod
    def equation_templates():
        """
        Predefined equation templates
        """
        templates = [
            # Linear model
            lambda G, a, b: a - b * G,
            
            # Michaelis-Menten type
            lambda G, a, b, c: a - b * G / (c + G),
            
            # Hill function type
            lambda G, a, b, c, n: a - b * G**n / (c**n + G**n),
            
            # With positive feedback
            lambda G, a, b, c, d: a - b * G + c * G**2 / (d**2 + G**2),
            
            # The form we derived earlier
            lambda G, a, b, c, d, n: a - b * G + c * G**n / (d**n + G**n),
            
            # With saturable consumption
            lambda G, a, b, c, d: a - b * G - c * G / (d + G)
        ]
        
        return templates
    
    def compute_derivative(self, t, G):
        """Compute numerical derivative"""
        dt = t[1] - t[0]
        dG = np.zeros_like(G)
        dG[1:-1] = (G[2:] - G[:-2]) / (2 * dt)
        dG[0] = (G[1] - G[0]) / dt
        dG[-1] = (G[-1] - G[-2]) / dt
        return dG
    
    def fitness(self, equation_func, params, t, G_data):
        """
        Calculate the goodness of fit for the equation
        """
        dG_data = self.compute_derivative(t, G_data)
        dG_pred = equation_func(G_data, *params)
        
        # Mean squared error
        mse = np.mean((dG_data - dG_pred)**2)
        
        # Add penalty for number of parameters (Occam's razor)
        complexity_penalty = 0.01 * len(params) * mse
        
        return mse + complexity_penalty
    
    def fit(self, t, G_data):
        """
        For a given time series data, find the best equation
        """
        templates = self.equation_templates()
        
        print("\nStarting symbolic regression...")
        print("=" * 50)
        
        results = []
        
        for i, template in enumerate(templates):
            print(f"\nTesting equation template {i+1}:")
            
            # Initial parameter guesses based on template
            if i == 0:  # Linear: a - b*G
                initial_params = [0.1, 0.01]
                bounds = [(0, 10), (0, 1)]
            elif i == 1:  # MM: a - b*G/(c+G)
                initial_params = [0.1, 0.5, 5.0]
                bounds = [(0, 10), (0, 5), (0, 20)]
            elif i == 2:  # Hill: a - b*G^n/(c^n+G^n)
                initial_params = [0.1, 0.5, 5.0, 2.0]
                bounds = [(0, 10), (0, 5), (0, 20), (1, 5)]
            elif i == 3:  # With positive feedback
                initial_params = [0.1, 0.01, 0.1, 5.0]
                bounds = [(0, 10), (0, 1), (0, 1), (1, 20)]
            elif i == 4:  # Our derived form
                initial_params = [0.1, 0.01, 0.1, 5.0, 2.0]
                bounds = [(0, 10), (0, 1), (0, 1), (1, 20), (1, 5)]
            else:  # Saturable consumption
                initial_params = [0.1, 0.01, 0.1, 5.0]
                bounds = [(0, 10), (0, 1), (0, 1), (1, 20)]
            
            # Optimize parameters
            def objective(params):
                try:
                    return self.fitness(template, params, t, G_data)
                except:
                    return float('inf')
            
            try:
                result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
                
                final_fitness = result.fun
                results.append((final_fitness, template, result.x, i))
                
                print(f"  Fitness: {final_fitness:.6f}")
                print(f"  Parameters: {result.x}")
                
            except Exception as e:
                print(f"  Optimization failed: {e}")
                results.append((float('inf'), template, None, i))
        
        # Select the best equation
        valid_results = [r for r in results if r[0] != float('inf')]
        if valid_results:
            best_idx = np.argmin([r[0] for r in valid_results])
            self.best_fitness, self.best_template, self.best_params, self.template_idx = valid_results[best_idx]
            
            print("\n" + "=" * 50)
            print(f"Best equation template: {self.template_idx + 1}")
            print(f"Best fitness: {self.best_fitness:.6f}")
            print(f"Best parameters: {self.best_params}")
        else:
            print("\nNo valid equations found!")
            self.best_template = None
            self.best_params = None
            self.template_idx = None
        
        return self.best_template, self.best_params
    
    def predict_derivative(self, G):
        """Predict derivative using the best found equation"""
        if self.best_template is None or self.best_params is None:
            raise ValueError("Must call fit() first")
        
        return self.best_template(G, *self.best_params)
    
    def simulate(self, G0, t):
        """Simulate time series using the discovered equation"""
        if self.best_template is None:
            raise ValueError("Must call fit() first")
        
        G_sim = np.zeros(len(t))
        G_sim[0] = G0
        
        for i in range(1, len(t)):
            dG = self.predict_derivative(G_sim[i-1])
            G_sim[i] = G_sim[i-1] + dG * (t[i] - t[i-1])
            G_sim[i] = max(0.1, G_sim[i])  # Prevent negative values
        
        return G_sim
    
    def get_equation_string(self):
        """Get a string representation of the discovered equation"""
        if self.best_template is None or self.best_params is None:
            return "No equation discovered"
        
        if self.template_idx == 0:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G"
        elif self.template_idx == 1:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G / ({self.best_params[2]:.4f} + G)"
        elif self.template_idx == 2:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G^{self.best_params[3]:.1f} / ({self.best_params[2]:.4f}^{self.best_params[3]:.1f} + G^{self.best_params[3]:.1f})"
        elif self.template_idx == 3:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G + {self.best_params[2]:.4f} * G^2 / ({self.best_params[3]:.4f}^2 + G^2)"
        elif self.template_idx == 4:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G + {self.best_params[2]:.4f} * G^{self.best_params[4]:.1f} / ({self.best_params[3]:.4f}^{self.best_params[4]:.1f} + G^{self.best_params[4]:.1f})"
        elif self.template_idx == 5:
            return f"dG/dt = {self.best_params[0]:.4f} - {self.best_params[1]:.4f} * G - {self.best_params[2]:.4f} * G / ({self.best_params[3]:.4f} + G)"
        else:
            return f"dG/dt = f(G) with parameters {self.best_params}"


# ============================================================================
# Part 4: Save Results to File
# ============================================================================

def save_results_to_file(model, scenarios, all_raw_data, regressor, 
                         t_train, G_train, t_test, G_test, G_simplified, G_simplified_test,
                         mse_train, mse_test, results_folder, latest_folder):
    """
    Save all results to the specified folders
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the discovered equation and analysis
    eq_filename = os.path.join(results_folder, f'discovered_equation.txt')
    eq_latest = os.path.join(latest_folder, f'discovered_equation.txt')
    
    with open(eq_filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GLUCOSE METABOLISM MODEL: DISCOVERED PARSIMONIOUS EQUATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DISCOVERED DIFFERENTIAL EQUATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{regressor.get_equation_string()}\n\n")
        
        f.write("PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        param_names = ['a', 'b', 'c', 'd', 'n'][:len(regressor.best_params)]
        for name, value in zip(param_names, regressor.best_params):
            f.write(f"{name} = {value:.6f}\n")
        f.write("\n")
        
        f.write("FIT STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training MSE: {mse_train:.6f}\n")
        f.write(f"Extrapolation MSE: {mse_test:.6f}\n")
        f.write(f"Fitness score: {regressor.best_fitness:.6f}\n\n")
        
        f.write("PHYSIOLOGICAL INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        f.write("- The constant term represents basal hepatic glucose output and/or food absorption\n")
        f.write("- The linear term in G represents insulin-independent consumption (e.g., brain)\n")
        f.write("- The nonlinear term represents insulin-dependent peripheral tissue uptake (muscle, adipose)\n")
        f.write("- The saturation term (K_m) reflects the limited capacity of transport proteins\n\n")
        
        f.write("DETAILED ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        if regressor.template_idx == 2:  # Hill equation
            a, b, c, n = regressor.best_params
            f.write(f"Equation form: dG/dt = {a:.4f} - {b:.4f} * G^{n:.1f} / ({c:.4f}^{n:.1f} + G^{n:.1f})\n")
            f.write(f"Half-saturation constant (K_m): {c:.2f} mmol/L\n")
            f.write(f"Hill coefficient: {n:.2f} (indicates {'positive' if n>1 else 'negative'} cooperativity)\n\n")
        
        f.write("COMPARISON WITH KNOWN PHYSIOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("- Normal fasting glucose: 4.0-5.5 mmol/L\n")
        f.write("- Typical K_m for GLUT transporters: 5-10 mmol/L\n")
        f.write("- Brain glucose uptake K_m: ~1 mmol/L (high affinity)\n")
        f.write("- Muscle GLUT4 K_m: ~5 mmol/L (insulin-dependent)\n")
    
    # Copy to latest folder
    with open(eq_filename, 'r') as src, open(eq_latest, 'w') as dst:
        dst.write(src.read())
    
    print(f"  Equation saved to: {eq_filename}")
    
    # 2. Save scenario data
    data_filename = os.path.join(results_folder, f'scenario_data.json')
    data_latest = os.path.join(latest_folder, f'scenario_data.json')
    
    with open(data_filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'scenarios': all_raw_data,
            'model_parameters': {
                'V_blood': model.V_blood,
                'V_liver': model.V_liver,
                'V_muscle': model.V_muscle,
                'V_fat': model.V_fat,
                'V_brain': model.V_brain,
                'insulin_sensitivity': model.insulin_sensitivity,
                'basal_insulin': model.basal_insulin
            }
        }, f, indent=2)
    
    # Copy to latest folder
    with open(data_filename, 'r') as src, open(data_latest, 'w') as dst:
        dst.write(src.read())
    
    print(f"  Scenario data saved to: {data_filename}")
    
    # 3. Save comparison data
    comparison_filename = os.path.join(results_folder, f'comparison_data.txt')
    comparison_latest = os.path.join(latest_folder, f'comparison_data.txt')
    
    with open(comparison_filename, 'w') as f:
        f.write("COMPLETE MODEL VS PARSIMONIOUS EQUATION COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TRAINING SCENARIO (Normal Meal):\n")
        f.write("-" * 30 + "\n")
        f.write("Time(min)\tComplete Model\tParsimonious Eq\n")
        for i in range(0, len(t_train), 10):  # Sample every 10th point
            f.write(f"{t_train[i]:.1f}\t{G_train[i]:.4f}\t{G_simplified[i]:.4f}\n")
        
        f.write("\nEXTRAPOLATION SCENARIO (OGTT):\n")
        f.write("-" * 30 + "\n")
        f.write("Time(min)\tComplete Model\tParsimonious Eq\n")
        for i in range(0, len(t_test), 10):
            f.write(f"{t_test[i]:.1f}\t{G_test[i]:.4f}\t{G_simplified_test[i]:.4f}\n")
    
    # Copy to latest folder
    with open(comparison_filename, 'r') as src, open(comparison_latest, 'w') as dst:
        dst.write(src.read())
    
    print(f"  Comparison data saved to: {comparison_filename}")
    
    return eq_filename, data_filename, comparison_filename


# ============================================================================
# Part 5: Main Program - Run Complete Workflow
# ============================================================================

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    print("=" * 60)
    print("Glucose Metabolic Network: From Multi-Compartment Model to AI-Discovered Parsimonious Equation")
    print("=" * 60)
    
    # 1. Initialize complete model
    model = GlucoseMetabolismModel()
    
    # 2. Define scenarios
    scenarios = [
        # Scenario 1: Normal meal response (breakfast + lunch)
        ("Normal meal", [60, 240], [30, 40]),  # 30g at 60min, 40g at 240min
        
        # Scenario 2: Oral Glucose Tolerance Test
        ("OGTT", [0], [75]),  # 75g glucose load
        
        # Scenario 3: Continuous feeding
        ("Continuous feeding", [0, 120, 240, 360], [20, 20, 20, 20]),
        
        # Scenario 4: Post-exercise meal (for testing extrapolation)
        ("Post-exercise", [180], [50]),
    ]
    
    # 3. Generate data
    print("\n[1/4] Generating complete model data...")
    all_t, all_G, labels, all_raw_data = generate_training_data(model, scenarios)
    
    # 4. Visualize original data
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for t, G, label in zip(all_t, all_G, labels):
        plt.plot(t, G, label=label, linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Blood Glucose (mmol/L)')
    plt.title('Complete Model: Blood Glucose Dynamics Under Different Scenarios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Discover parsimonious equation using symbolic regression
    print("\n[2/4] Discovering parsimonious equation with symbolic regression...")
    
    # Use data from first scenario (normal meal) for training
    t_train = all_t[0]
    G_train = all_G[0]
    
    regressor = SymbolicRegressor(population_size=30, generations=15)
    best_template, best_params = regressor.fit(t_train, G_train)
    
    # 6. Validate the discovered equation
    print("\n[3/4] Validating discovered equation...")
    
    plt.subplot(2, 2, 2)
    
    # Derivative of original data
    dG_data = regressor.compute_derivative(t_train, G_train)
    plt.scatter(G_train, dG_data, alpha=0.5, label='Data derivatives', s=10)
    
    # Predicted derivative
    G_range = np.linspace(4, 12, 100)
    dG_pred = regressor.predict_derivative(G_range)
    plt.plot(G_range, dG_pred, 'r-', linewidth=3, label='Discovered equation')
    
    plt.xlabel('Blood Glucose G (mmol/L)')
    plt.ylabel('dG/dt (mmol/L/min)')
    plt.title('Phase Space: dG/dt vs G')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Compare complete model vs parsimonious equation time series
    plt.subplot(2, 2, 3)
    
    # Simulate with discovered equation
    G0 = G_train[0]
    G_simplified = regressor.simulate(G0, t_train)
    
    plt.plot(t_train, G_train, 'b-', linewidth=2, label='Complete model')
    plt.plot(t_train, G_simplified, 'r--', linewidth=2, label='Parsimonious equation')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Blood Glucose (mmol/L)')
    plt.title('Complete Model vs Parsimonious Equation (Training Scenario)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate error
    mse_train = np.mean((G_train - G_simplified)**2)
    print(f"Training scenario MSE: {mse_train:.4f}")
    
    # 8. Test extrapolation ability (OGTT scenario)
    plt.subplot(2, 2, 4)
    
    t_test = all_t[1]  # OGTT scenario
    G_test = all_G[1]
    
    # Simulate OGTT with discovered equation
    G0_test = G_test[0]
    G_simplified_test = regressor.simulate(G0_test, t_test)
    
    plt.plot(t_test, G_test, 'b-', linewidth=2, label='Complete model (OGTT)')
    plt.plot(t_test, G_simplified_test, 'r--', linewidth=2, label='Parsimonious prediction')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Blood Glucose (mmol/L)')
    plt.title('Extrapolation Test: OGTT Scenario')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    mse_test = np.mean((G_test - G_simplified_test)**2)
    print(f"Extrapolation scenario MSE: {mse_test:.4f}")
    
    plt.tight_layout()
    
    # 9. Print discovered equation
    print("\n[4/4] Discovered Parsimonious Equation")
    print("=" * 50)
    
    eq_str = regressor.get_equation_string()
    print(f"\nDiscovered differential equation:")
    print(f"\033[1m{eq_str}\033[0m")
    print(f"\nParameter values: {regressor.best_params}")
    
    # 10. Physiological interpretation
    print("\nPhysiological interpretation:")
    print("- The constant term represents basal hepatic glucose output and/or food absorption")
    print("- The linear term in G represents insulin-independent consumption (e.g., brain)")
    print("- The nonlinear term represents insulin-dependent peripheral tissue uptake (muscle, adipose)")
    print("- The saturation term (K_m) reflects the limited capacity of transport proteins")
    
    # 11. Additional analysis
    print("\n" + "=" * 50)
    print("ANALYSIS OF DISCOVERED EQUATION")
    print("=" * 50)
    
    if regressor.best_params is not None and len(regressor.best_params) >= 4 and regressor.template_idx == 2:
        # For the Hill-type equation (template 3, index 2)
        a, b, c, n = regressor.best_params[:4]
        
        print(f"\nThe discovered equation: dG/dt = {a:.4f} - {b:.4f} * G^{n:.1f} / ({c:.4f}^{n:.1f} + G^{n:.1f})")
        print("\nKey insights:")
        
        # Steady state analysis
        print("- Steady state occurs when dG/dt = 0")
        
        # Calculate approximate steady state
        if a > 0 and b > 0:
            try:
                # For Hill equation, solve a = b * G^n/(c^n+G^n)
                # => G^n/(c^n+G^n) = a/b
                # => G^n = (a/b) * (c^n+G^n)
                # => G^n * (1 - a/b) = (a/b) * c^n
                # => G^n = (a/b) * c^n / (1 - a/b)
                ratio = a/b
                if ratio < 1:
                    G_ss = c * (ratio/(1-ratio))**(1/n)
                    if 0 < G_ss < 20:
                        print(f"- Predicted fasting glucose: ~{G_ss:.2f} mmol/L")
            except:
                pass
        
        # Saturation analysis
        print(f"- Half-saturation constant (K_m): {c:.2f} mmol/L")
        print(f"- Hill coefficient: {n:.2f} (indicates {'positive' if n>1 else 'negative'} cooperativity)")
        
        print("\nComparison with known physiology:")
        print("- Normal fasting glucose: 4.0-5.5 mmol/L")
        print("- Typical K_m for GLUT transporters: 5-10 mmol/L")
        print("- Brain glucose uptake K_m: ~1 mmol/L (high affinity)")
        print("- Muscle GLUT4 K_m: ~5 mmol/L (insulin-dependent)")
    
    # 12. Create results folder and save all results
    print("\n" + "=" * 50)
    print("SAVING RESULTS TO FOLDER")
    print("=" * 50)
    
    # Create results folders
    results_folder, latest_folder = create_results_folder()
    print(f"Creating results folder: {results_folder}")
    print(f"Creating latest folder: {latest_folder}")
    
    # Save the figure to both folders
    figure_filename = os.path.join(results_folder, f'glucose_model_comparison.png')
    plt.savefig(figure_filename, dpi=150, bbox_inches='tight')
    
    # Also save to latest folder
    latest_figure = os.path.join(latest_folder, f'glucose_model_comparison.png')
    plt.savefig(latest_figure, dpi=150, bbox_inches='tight')
    
    print(f"  Figure saved to: {figure_filename}")
    
    # Save all data files
    eq_file, data_file, comp_file = save_results_to_file(
        model, scenarios, all_raw_data, regressor,
        t_train, G_train, t_test, G_test, 
        G_simplified, G_simplified_test,
        mse_train, mse_test,
        results_folder, latest_folder
    )
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"All results saved to:")
    print(f"  Timestamped folder: {results_folder}")
    print(f"  Latest folder: {latest_folder}")
    print("\nFiles created:")
    print(f"  1. glucose_model_comparison.png")
    print(f"  2. discovered_equation.txt")
    print(f"  3. scenario_data.json")
    print(f"  4. comparison_data.txt")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
