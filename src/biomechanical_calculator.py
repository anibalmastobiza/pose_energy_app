"""
Biomechanical calculations for energy expenditure and protein requirements
Based on peer-reviewed scientific literature
"""
import numpy as np

class BiomechanicalCalculator:
    """
    Implements biomechanically optimal formulas for energy expenditure
    and protein synthesis requirements
    """
    
    def __init__(self, weight_kg=70, height_cm=170, age=30, sex='male'):
        """
        Initialize calculator with user anthropometric data
        
        Args:
            weight_kg: Body weight in kilograms
            height_cm: Height in centimeters
            age: Age in years
            sex: Biological sex ('male' or 'female')
        """
        self.weight = weight_kg
        self.height = height_cm
        self.age = age
        self.sex = sex
        self.lean_mass = self.estimate_lean_mass()
        
    def estimate_lean_mass(self):
        """
        Estimate lean body mass using Boer formula (1984)
        
        Returns:
            float: Estimated lean body mass in kg
        
        Reference:
            Boer P. (1984). Am J Physiol. 247(4):F632-5
        """
        if self.sex == 'male':
            return (0.407 * self.weight) + (0.267 * self.height) - 19.2
        else:
            return (0.252 * self.weight) + (0.473 * self.height) - 48.3
    
    def calculate_energy_expenditure(self, velocity_data, acceleration_data, duration_s):
        """
        Calculate total energy expenditure in Joules using biomechanical principles
        
        Args:
            velocity_data: Array of velocity magnitudes (m/s)
            acceleration_data: Array of acceleration vectors (m/s²)
            duration_s: Duration of activity in seconds
            
        Returns:
            float: Total energy expenditure in Joules
            
        References:
            - Cavagna & Kaneko (1977). J Physiol. 268(2):467-81
            - Willems et al. (1995). J Exp Biol. 198:379-393
        """
        # Kinetic energy changes (J)
        ke_changes = 0.5 * self.weight * np.sum(np.diff(velocity_data**2))
        
        # Work against gravity - vertical displacement (J)
        if len(acceleration_data.shape) > 1:
            vertical_work = self.weight * 9.81 * np.sum(np.abs(np.diff(acceleration_data[:, 1])))
        else:
            vertical_work = self.weight * 9.81 * np.sum(np.abs(acceleration_data)) * 0.1
        
        # Internal work for limb movements - Willems et al. (1995)
        # Approximately 10% of total work for human locomotion
        internal_work = 0.1 * self.weight * np.sum(np.abs(acceleration_data))
        
        # Total mechanical work (J)
        total_mechanical_work = abs(ke_changes) + vertical_work + internal_work
        
        # Convert to metabolic energy using efficiency factor
        # Human movement efficiency: 20-25% (Cavagna & Kaneko, 1977)
        metabolic_efficiency = 0.23
        metabolic_energy = total_mechanical_work / metabolic_efficiency
        
        # Add basal metabolic rate component for activity duration
        bmr_joules_per_second = self.calculate_bmr() * 4.184 / 86400  # Convert kcal/day to J/s
        bmr_component = bmr_joules_per_second * duration_s
        
        return metabolic_energy + bmr_component
    
    def calculate_bmr(self):
        """
        Calculate Basal Metabolic Rate using Cunningham equation (1991)
        Most accurate for athletic populations
        
        Returns:
            float: BMR in kcal/day
            
        Reference:
            Cunningham JJ. (1991). Am J Clin Nutr. 54(6):963-9
        """
        return 500 + (22 * self.lean_mass)
    
    def estimate_protein_needs(self, energy_joules, activity_intensity):
        """
        Calculate optimal protein intake for recovery and adaptation
        
        Args:
            energy_joules: Total energy expenditure in Joules
            activity_intensity: METs value indicating exercise intensity
            
        Returns:
            float: Recommended protein intake in grams
            
        References:
            - Jäger et al. (2017). ISSN Position Stand. J Int Soc Sports Nutr. 14:20
            - Moore et al. (2015). J Appl Physiol. 119(3):290-301
            - Witard et al. (2014). Am J Clin Nutr. 99(1):86-95
        """
        # Convert energy to kcal
        energy_kcal = energy_joules / 4184
        
        # Determine protein factor based on activity intensity (g/kg/day)
        # Based on ISSN Position Stand (2017)
        if activity_intensity < 3:  # Light activity
            protein_factor = 0.8
        elif activity_intensity < 6:  # Moderate activity
            protein_factor = 1.2
        elif activity_intensity < 9:  # Vigorous activity
            protein_factor = 1.6
        else:  # Very vigorous activity
            protein_factor = 2.0
        
        # Calculate session-specific protein needs
        # Scaled by energy expenditure relative to daily energy
        daily_energy_estimate = 2000  # kcal (average)
        session_fraction = energy_kcal / daily_energy_estimate
        session_protein = protein_factor * self.weight * session_fraction
        
        # Apply minimum effective dose for muscle protein synthesis
        # Moore et al. (2015): 20-25g optimal range
        # Witard et al. (2014): 2.5g leucine threshold (~20g high-quality protein)
        minimum_effective_dose = 20  # grams
        maximum_single_dose = 40  # grams (plateau of MPS response)
        
        return np.clip(session_protein, minimum_effective_dose, maximum_single_dose)
    
    def calculate_met_value(self, velocity_magnitude):
        """
        Estimate METs from movement velocity
        
        Args:
            velocity_magnitude: Average velocity in m/s
            
        Returns:
            float: Estimated METs value
            
        Reference:
            Ainsworth et al. (2011). Med Sci Sports Exerc. 43(8):1575-81
        """
        # Velocity to METs conversion based on Compendium of Physical Activities
        if velocity_magnitude < 0.5:
            return 2.0  # Light activity
        elif velocity_magnitude < 1.0:
            return 3.5  # Light-moderate
        elif velocity_magnitude < 1.5:
            return 5.0  # Moderate
        elif velocity_magnitude < 2.0:
            return 7.0  # Vigorous
        elif velocity_magnitude < 2.5:
            return 9.0  # Very vigorous
        else:
            return 11.0  # Near maximal
