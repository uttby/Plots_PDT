from scipy import constants as sc

class ExperimentalSetup:
    def __init__(self, epsilon_PpIX, epsilon_Ppp, c__oxy, S__t0_PpIX, wavelength, power_density, irradiation_time):
        """
        This object manages all experimental condition dependent parameters of the PDT model. 

        Args:
            epsilon_PpIX (pandas.core.series.Series): Absorption coefficient of PpIX (for a range of wavelenghts).
            epsilon_Ppp (pandas.core.series.Series): Absorption coefficient of Ppp (for a range of wavelenghts).
            c__oxy (float): Concentration of oxygen in the tissue.
            S__t0_PpIX (float): Initial concentration of PpIX.
            wavelength (float): Wavelength of irradiation light (in nanometers).
            power_density (float): Power density of the irradiation light (in mW/cm^2).
        """
        self.epsilon_PpIX = epsilon_PpIX
        self.epsilon_Ppp = epsilon_Ppp
        self.c__oxy = c__oxy
        self.S__t0_PpIX = S__t0_PpIX
        self.wavelength = wavelength
        self.power_density = power_density
        self.irradiation_time = irradiation_time

    # Getter methods
    def get_c__oxy(self):
        return self.c__oxy

    def get_S__t0_PpIX(self):
        return self.S__t0_PpIX

    def get_irradiation_time(self):
        return self.irradiation_time

    def get_wavelength(self):
        return self.wavelength
    
    def get_power_density(self):
        return self.power_density
    
    # Setter methods
    def set_c__oxy(self, c__oxy):
        self.c__oxy = c__oxy

    def set_S__t0_PpIX(self, S__t0_PpIX):
        self.S__t0_PpIX = S__t0_PpIX

    def set_irradiation_time(self, irradiation_time):
        self.irradiation_time = irradiation_time

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength

    def set_power_density(self, power_density):
        self.power_density = power_density

    def get_EPR(self):
        """
        Returns the emitted photon rate for the stored experimental setup.
        """
        # Absorbed photon rate 
        photon_energy = sc.Planck * sc.speed_of_light / (self.wavelength * pow(10, -9))    
        # [m^2 kg / s][m / s][1 / m] -> J

        # Emitted photon rate 
        EPR = (self.power_density * pow(10, -3)) * (1 / photon_energy)  # [W/cm^2][1/J]  -> [1/(s*cm^2)]
        return EPR

    def get_APR_PpIX(self):
        """
        Returns the absorbed photon rate for PpIX by calculating it using 
        the stored irradiation power density and wavelenght.
        """
    
        absorbance_cross_section_PpIX = self.epsilon_PpIX * (2.3025 * 1000 / sc.Avogadro)
        return (absorbance_cross_section_PpIX.at[self.wavelength] * self.get_EPR())  # -> [1/s]
    
    def get_APR_Ppp(self):
        """
        Returns the absorbed photon rate for Ppp by calculating it using 
        the stored irradiation power density and wavelenght.
        """
        
        absorbance_cross_section_Ppp = self.epsilon_Ppp * (2.3025 * 1000 / sc.Avogadro)
        return (absorbance_cross_section_Ppp.at[self.wavelength] * self.get_EPR())  # -> [1/s]