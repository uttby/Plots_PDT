import pandas as pd
import sys

sys.path.append('../')

from model.parameter_setup.experimental_setup import ExperimentalSetup
from model.parameter_setup.specific_parameter_setup import SpecificParameterSetup
from model.parameter_setup.visualization_setup import VisualizationSetup

"""
Configuration file. 
If there are changes in experimetal values or mass spectrometry file naming, 
just change the values in this configuration file and they will be altered in the entire code.
"""

# ---------------------------------ABSORBANCE DATA PREPARATION AND ABSORBANCE COEFFICIENT CALCULATION-----------------------------------
# Data location of an absorbance intensity measurement with 10 mW/cm^2 and a Energy Dose of 0J and 100J 
intensity_100J_0J_location = "data_files/absorbance_PpIX_635nm_10mW_0J_100J.csv"

# Read the data 
intensity_100J_0J = pd.read_csv(intensity_100J_0J_location, sep = ",").astype(float).astype({"wavelenght" : "int"}).set_index("wavelenght")

# Initial concentration of PpIX for the absorbance intensity measurement
concentration_PpIX_0J =  10e-6 # mol/l

# Concentration of PpIX and Ppp after the solution has been irradiated with an energy Dose of 100J 
concentration_PpIX_100J = 2.382e-6 # mol/l
concentration_Ppp_100J =  0.986e-6 # mol/l

# The lenght the light travels inside the solution while the absorbance intensity is measured.
L = 1 #cm

# Beer-Lambert-Law:
# ε = A/Lc 
    
# A is the amount of light absorbed by the sample for a particular wavelength
# ε is the molar extinction coefficient L/(mol*cm)
# L is the distance that the light travels through the solution
# c is the concentration of the absorbing species per unit volume

# Calculate the apsorption coefficient for PpIX
epsilonPpIX = intensity_100J_0J["0J"] * 1/(concentration_PpIX_0J) * 1/L

# Calculate the apsorption coefficient for Ppp using the total intensity at 100J and subtract the influence of PpIX 
epsilonPpp = (intensity_100J_0J["100J"]/L - epsilonPpIX * concentration_PpIX_100J) / (concentration_Ppp_100J)

# ---------------------------------EXPERIMENTAL SETUP DEFINITIONS--------------------------------------------------
# Create an instance of ExperimentalSetup with all the experimental conditions for the simulation
experimental_setup = ExperimentalSetup(
    # absorption coefficient of PpIX in [L/(mol*cm)]=[1/(cm*M)] (calculated above)
    epsilon_PpIX=epsilonPpIX, 
    # absorption coefficient of Ppp in [L/(mol*cm)]=[1/(cm*M)] (calculated above)
    epsilon_Ppp=epsilonPpp, 
    # ground state oxygen concentration (assumed to be constant) in μM
    c__oxy=40, 
    # starting concentration of PpIX in μM
    S__t0_PpIX= 10, 
    # irradiation wavelenght in nm
    wavelength=635, 
    # irradiation power density mW/cm2
    power_density = 100,
    # irradiation time used in the visualization plots
    irradiation_time = 100)

# ---------------------------------SPECIFIC SETUP DEFINITIONS--------------------------------------------------
# Create an instance of SpecificParameterSetup with the specified parameter values for the simulation
specific_parameter_setup = SpecificParameterSetup(
    # ξ symbolyses the specific oxygen consuption rate 
    # ξ'=S_Δ ϕ_t *(k_oa [A])/(k_d + k_oa[A])*σ/hv
    xi_PpIX=0.021, # [mJ/cm^2]
    xi_Ppp=0.021,

    # β = k_p/k_ot 
    # k_p : phosphorescence rate
    # k_ot : rate of reaction between ground state oxygen and triplet state photosensitizer (creating singlet oxygen)
    beta_PpIX=4,
    beta_Ppp=4,

    # μ(pwr_den) = k_os(pwr_den)/(k_oa*[A])"),
    # k_os : rate of reaction between singlet oxygen and ground state photosensitizer (photobleaching), is assumed to be power density dependent in this work. 
    # k_oa : rate of reaction between singlet oxygen and acceptor (cancer cell)
    # [A] : Acceptor cell concentration
    mu_PpIX=9.1e-5,
    mu_Ppp=9.1e-5,

    # δ : Low concentration correction term. If the concentration of photosensitizer is low, the probability is higher that 
    # the photosensitizer molecule will react with the singlet oxygen it has created. This probability is included into the model
    # via the low concentration correction term δ.
    delta_PpIX=33,
    delta_Ppp=33,

    # Probabililty of singlet excited state photosensitizer to move into triplet excited state.
    # ϕ_t = k_isc / (km + k_isc)
    # k_isc: rate of intersystem crossing
    # k_m: rate of decay from singlet excited state to ground state (radiative and non radiative)
    Phi_t_PpIX=0.831,  
    Phi_t_Ppp=0.831,

    # The fraction of triplet state photosensitizer quenching by ground state oxygen that results in the production of singlet oxygen.
    # can be aqquired using the singlet oxygen quantum yield with the following equation:
    # S_Delta = quantum_yield * (beta + gamma + c__oxy) / (Phi__t * c__oxy)
    # Using the quantum yield of 0.7 and values for beta, gamma, Phi__t and c__oxy gives the following equation for S_Delta:
    S_Delta_PpIX=round(0.7 * (4 + 0 + 80) / (0.831 * 80),4),
    S_Delta_Ppp=round(0.7 * (4 + 0 + 80) / (0.831 * 80),4), 

    # γ = k_ta[A]/k_ot (= 0, if Type I reaction is assumed to be neglectable), 
    # k_ta : type I reaction rate (rate of reaction between triplet state photosensitizer and acceptor cell)
    # [A]  : acceptor cell concentration
    # k_ot : rate of reaction between ground state oxygen and triplet state photosensitizer (creating singlet oxygen)
    gamma_PpIX=0, # assuming Type I reaction neglectable
    gamma_Ppp=0   # assuming Type I reaction neglectable
)

# ---------------------------------VISUALIZATION SETUP DEFINITIONS--------------------------------------------------
# Define visualization labels
TIME_LABEL = "Time in s"
ENERGY_DOSE_LABEL = "Energy Dose in J/cm^2"

# Create an instance of VisualizationSetup
visualization_setup = VisualizationSetup(
    x_label=TIME_LABEL,
    x_factor=1
)

# --------------------------------MASS SPECTRONMETRY SETUP CONSTANTS-------------------------------------------------
# The uploadable mass spectrometry data file including 
# only the solvent data is supposed to be named the following way: 
solvent_tag = "SOLV"

# The uploadable mass spectrometry data is supposed to be named 
# "[Energy Density (numeral)] + J"
data_file_naming_specification = r"(\d+)J"

# Relationship between concentration and intensity as described in Sochis paper  TODO real referece
# Intensity = 13.2 + 101.2 * concentration (in uM)
# Assuming same relationsship between absorbance intensity and concentration for PpIX and Ppp
factor_PpIX = 101.2
bias_PpIX = 13.2
element_mass_PpIX = 563.5 #TODO 563.658
#10mW
#   01: 563.92 
#   02: 563.38 
#   03: 563.44 -
#   04: 563.48
#50mW
#   01: 563.48
#   02: 563.91 
#   03: 563.93 
#   04: 563.42
#100mW
#   01: 563.90
#   02: 563.94
#   03: 563.93 -
#   04: 563.42
#   05: 563.87

factor_Ppp = 101.2
bias_Ppp = 13.2
element_mass_Ppp = 595.5 #TODO 595.657 
#10mW
#   01: 595.94 -
#   02: 595.4
#   03: 595.45
#   04: 595.47
#50mW
#   01: 595.47
#   02: 595.96
#   03: 595.95 -
#   04: 595.43
#100mW
#   01: 595.95
#   02: 595.93
#   03: 595.96 -
#   04: 595.42
#   05: 595.95

interval_length = 2;

# Create an empty dictionary that is used to store the uploaded data.
uploaded_data = {}
