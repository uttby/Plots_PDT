import numpy as np
from scipy.integrate import odeint
from config import experimental_setup, specific_parameter_setup

def ground_state_concentration(t, mu_PpIXn, mu_Pppn):
    """ 
    Numerically solves for the ground state concentration of both PpIX and Ppp.
        Args:
            t(List): time in discrete steps

        Returns([List, List]): 
            numerical solution for the ground state concentration for PpIX and Ppp
    """

    # Extract specific parameters for PpIX and Ppp from the specific parameter setup
    xi_dash_PpIX, xi_dash_Ppp, beta_PpIX, beta_Ppp, mu_PpIX, mu_Ppp, delta_PpIX, delta_Ppp, \
    Phi__PpIX_t, Phi__Ppp_t, S_Delta_PpIX, S_Delta_Ppp, gamma_PpIX, gamma_Ppp = specific_parameter_setup.get_all_parameters()
    mu_PpIX = mu_PpIXn
    mu_Ppp = mu_Pppn
    # Extract experimental parameters for PpIX and Ppp from the experimental setup
    power_density = experimental_setup.get_power_density()
    c__oxy = experimental_setup.get_c__oxy()
    S__t0_PpIX = experimental_setup.get_S__t0_PpIX()
    APR_PpIX, APR_Ppp = experimental_setup.get_APR_PpIX(), experimental_setup.get_APR_Ppp()    
    # Define the function that represents the system of differential equations
    # (for more information about how these equations are formed, please refer to the supporting information)
    def system(y, t):
        S__0_PpIX, S__0_Ppp = y
   # Define the function that represents the system of differential equations
    # (for more information about how these equations are formed, please refer to the supporting information)
    def system(y, t):
        S__0_PpIX, S__0_Ppp = y

        dS__0_PpIX_dt = -mu_PpIX*c__oxy/2 * (
            S__0_PpIX * (xi_dash_Ppp * power_density * S__0_Ppp)/(beta_Ppp + gamma_Ppp + c__oxy)
            + 
            (S__0_PpIX + delta_PpIX)*(xi_dash_PpIX * power_density * S__0_PpIX)/(beta_PpIX + gamma_PpIX + c__oxy)
        )
        # dS__0_PpIX_dt = -(c__oxy * mu_PpIX * xi_dash_PpIX * power_density * (beta_Ppp + gamma_Ppp + c__oxy) * S__0_PpIX
        #     + c__oxy * power_density * mu_PpIX * xi_dash_Ppp * (beta_PpIX + gamma_PpIX + c__oxy) * S__0_PpIX
        #     + (c__oxy * delta_PpIX * mu_PpIX * power_density * xi_dash_PpIX + 2 * Phi__PpIX_t * mu_PpIX * APR_PpIX)
        #     * (beta_Ppp + gamma_Ppp + c__oxy)) * S__0_PpIX / ((beta_Ppp + gamma_Ppp + c__oxy)*(beta_PpIX + gamma_PpIX + c__oxy) * 2)
        dS__0_Ppp_dt = -mu_Ppp*c__oxy/2 * (
            S__0_Ppp * (xi_dash_PpIX * power_density * S__0_PpIX)/(beta_PpIX + gamma_PpIX + c__oxy)
            + 
            (S__0_Ppp + delta_Ppp)*(xi_dash_Ppp * power_density * S__0_Ppp)/(beta_Ppp + gamma_Ppp + c__oxy)) + mu_PpIX*c__oxy/2 * (
            S__0_PpIX * (xi_dash_Ppp * power_density * S__0_Ppp)/(beta_Ppp + gamma_Ppp + c__oxy)
            + 
            (S__0_PpIX + delta_PpIX)*(xi_dash_PpIX * power_density * S__0_PpIX)/(beta_PpIX + gamma_PpIX + c__oxy)
        )

        # dS__0_Ppp_dt = 1/(2*(beta_PpIX + gamma_PpIX + c__oxy)*(beta_Ppp + gamma_Ppp + c__oxy)) * (
        #     -c__oxy * mu_Ppp * xi_dash_Ppp * power_density * (beta_PpIX + gamma_PpIX + c__oxy) * S__0_Ppp * S__0_Ppp
        #     + (
        #         (
        #             (-mu_Ppp * xi_dash_PpIX + mu_PpIX * xi_dash_Ppp) * c__oxy * power_density 
        #             + mu_PpIX * xi_dash_Ppp * power_density * (beta_PpIX + gamma_PpIX) 
        #             - mu_Ppp * xi_dash_PpIX * power_density * (beta_Ppp + gamma_Ppp)
        #         ) * c__oxy * S__0_PpIX 
        #         - (c__oxy * delta_Ppp * mu_Ppp * power_density * xi_dash_Ppp + APR_Ppp * 2 * Phi__Ppp_t * gamma_Ppp) * (beta_PpIX + gamma_PpIX + c__oxy)
        #     ) * S__0_Ppp 
        #     + c__oxy * S__0_PpIX * mu_PpIX * xi_dash_PpIX * power_density * (S__0_PpIX + delta_PpIX) * (beta_Ppp + gamma_Ppp + c__oxy))

        return [dS__0_PpIX_dt, dS__0_Ppp_dt]

    # Define the initial conditions
    y0 = [S__t0_PpIX, 0]

    # Solve the system of differential equations numerically
    solution = odeint(system, y0, t)
    S__0_PpIX = solution[:, 0]
    S__0_Ppp  = solution[:, 1]

    # Return the nurmerical solutions for PpIX and Ppp
    return [S__0_PpIX, S__0_Ppp]

def singlet_oxygen_concentration(t, mu_PpIXn, mu_Pppn):
    """ 
    Uses the numerical solution for the ground state concentration 
    to solve for the singlet oxygen concentration multiplied with koaA for each and both PpIX and Ppp.
        Args:
            t(List): time in discrete steps

        Returns([List, List, List]): 
            singlet oxygen concentration produced from each PpIX and Ppp as well as the total amount (sum)
    """

    # Extract specific parameters for PpIX and Ppp from the specific_parameter_setup object
    xi_dash_PpIX, xi_dash_Ppp, beta_PpIX, beta_Ppp, mu_PpIX, mu_Ppp, delta_PpIX, delta_Ppp, \
    Phi__PpIX_t, Phi__Ppp_t, S_Delta_PpIX, S_Delta_Ppp, gamma_PpIX, gamma_Ppp = specific_parameter_setup.get_all_parameters()
    mu_PpIX = mu_PpIXn
    mu_Ppp = mu_Pppn

    # Extract experimental parameters for PpIX and Ppp from the experimental_setup object
    power_density = experimental_setup.get_power_density()
    c__oxy = experimental_setup.get_c__oxy()
    S__t0_PpIX = experimental_setup.get_S__t0_PpIX()

    # Define koaA as a constant (since we don't know it's value and therefore visualize the singlet oxygen concentration
    # multiplied by koaA 
    koaA = 1
    
    # Calculate ground state concentrations of PpIX and Ppp
    ground_state_PS = ground_state_concentration(t, mu_PpIXn, mu_Pppn)
    S__0_PpIX = ground_state_PS[0]
    S__0_Ppp = ground_state_PS[1]

    # Calculate singlet oxygen concentrations for PpIX and Ppp
    singlet_oxygen_PpIX = (c__oxy * xi_dash_PpIX * power_density * S__0_PpIX) \
                          / (2 * koaA * (beta_Ppp + gamma_Ppp + c__oxy))

    singlet_oxygen_Ppp = (c__oxy * xi_dash_Ppp * power_density * S__0_Ppp) \
                          / (2 * koaA * (beta_PpIX + gamma_PpIX + c__oxy))

    # Calculate the total singlet oxygen concentration (sum of PpIX and Ppp contributions)
    singlet_oxygen_total = singlet_oxygen_PpIX + singlet_oxygen_Ppp

    # Return the concentrations of singlet oxygen for PpIX, Ppp, and the total
    return [singlet_oxygen_PpIX, singlet_oxygen_Ppp, singlet_oxygen_total]

def emitted_singlet_oxygen(t, mu_PpIXn, mu_Pppn):
    """ 
    Uses the numerical solution for the ground state concentration 
    to solve for the integrated amount of the emitted singlet oxygen for each and both PpIX and Ppp.

        Args:
            t(List): time in discrete steps

        Returns([List, List, List]): 
            emitted singlet oxygen concentration produced from each PpIX and Ppp as well as the total amount (sum)
    """
        
    # Extract specific parameters for PpIX and Ppp from the specific_parameter_setup object
    xi_dash_PpIX, xi_dash_Ppp, beta_PpIX, beta_Ppp, mu_PpIX, mu_Ppp, delta_PpIX, delta_Ppp, \
    Phi__PpIX_t, Phi__Ppp_t, S_Delta_PpIX, S_Delta_Ppp, gamma_PpIX, gamma_Ppp = specific_parameter_setup.get_all_parameters()
    mu_PpIX = mu_PpIXn
    mu_Ppp = mu_Pppn
    # Extract experimental parameters for PpIX and Ppp from the experimental_setup object
    c__oxy = experimental_setup.get_c__oxy()
    APR_PpIX, APR_Ppp = experimental_setup.get_APR_PpIX(), experimental_setup.get_APR_Ppp()
    
    # Integrate the ground state concentration over time using cumulative sum
    delta_t = t[-1] / (len(t) - 1)
    ground_state_PS = ground_state_concentration(t, mu_PpIXn, mu_Pppn)
    int_S__0_PpIX_values = np.cumsum(ground_state_PS[0] * delta_t)
    int_S__0_Ppp_values = np.cumsum(ground_state_PS[1] * delta_t)

    # Calculate emitted singlet oxygen for PpIX and Ppp using integrated ground state concentrations
    emitted_singlet_oxygen_PpIX = (S_Delta_PpIX * c__oxy * Phi__PpIX_t * APR_PpIX) * int_S__0_PpIX_values / (beta_PpIX + gamma_PpIX + c__oxy)
    emitted_singlet_oxygen_Ppp = (S_Delta_Ppp * c__oxy * Phi__Ppp_t * APR_Ppp) * int_S__0_Ppp_values / (beta_Ppp + gamma_Ppp + c__oxy)

    # Calculate the total emitted singlet oxygen (sum of PpIX and Ppp contributions)
    emitted_singlet_oxygen_total = emitted_singlet_oxygen_PpIX + emitted_singlet_oxygen_Ppp

    # Return the emitted singlet oxygen concentrations for PpIX, Ppp, and the total
    return [emitted_singlet_oxygen_PpIX, emitted_singlet_oxygen_Ppp, emitted_singlet_oxygen_total]
    
def reactive_singlet_oxygen(t, mu_PpIXn, mu_Pppn):
    """ 
    Uses the numerical solution for the ground state concentration 
    to solve for integrated amount of singlet oxygen that is assumed to react with cancer cells for each and both PpIX and Ppp.

        Args:
            t(List): time in discrete steps

        Returns([List, List, List]): 
            reactive singlet oxygen concentration produced from each PpIX and Ppp as well as the total amount (sum)
    """

    # Assuming gamma_Ppp and gamma_PpIX to be zero (refer to supporting information for more information)

    # Extract specific parameters for PpIX and Ppp from the specific_parameter_setup object
    xi_dash_PpIX, xi_dash_Ppp, beta_PpIX, beta_Ppp, mu_PpIX, mu_Ppp, delta_PpIX, delta_Ppp, \
    Phi__PpIX_t, Phi__Ppp_t, S_Delta_PpIX, S_Delta_Ppp, gamma_PpIX, gamma_Ppp = specific_parameter_setup.get_all_parameters()

    # Extract experimental parameters for PpIX and Ppp from the experimental_setup object
    power_density = experimental_setup.get_power_density()
    c__oxy = experimental_setup.get_c__oxy()
    S__t0_PpIX = experimental_setup.get_S__t0_PpIX()
    APR_PpIX, APR_Ppp = experimental_setup.get_APR_PpIX(), experimental_setup.get_APR_Ppp()
    
    mu_PpIX = mu_PpIXn
    mu_Ppp = mu_Pppn

    # integrate the ground state concentration over time using cumulative sum
    delta_t = t[-1] / (len(t) - 1)
    
    ground_state_PS = ground_state_concentration(t, mu_PpIXn, mu_Pppn)
    int_S__0_PpIX_values = np.cumsum(ground_state_PS[0] * delta_t)
    int_S__0_Ppp_values = np.cumsum(ground_state_PS[1] * delta_t)
    print (f"delta_t {delta_t}")

    # Calculate reactive singlet oxygen for PpIX and Ppp using integrated ground state concentrations
    factor_PpIX = (c__oxy * xi_dash_PpIX * power_density) / (2*(beta_PpIX + c__oxy))
    factor_Ppp = (c__oxy * xi_dash_Ppp * power_density) / (2*(beta_Ppp + c__oxy))

    emitted_singlet_oxygen_PpIX = factor_PpIX * int_S__0_PpIX_values
    emitted_singlet_oxygen_Ppp  = factor_Ppp * int_S__0_Ppp_values
    print (f"factor {factor_PpIX}, {factor_Ppp}")
    # Calculate the total reactive singlet oxygen (sum of PpIX and Ppp contributions)
    emitted_singlet_oxygen_total = emitted_singlet_oxygen_PpIX + emitted_singlet_oxygen_Ppp

    # Return the reactive singlet oxygen concentrations for PpIX, Ppp, and the total
    return [emitted_singlet_oxygen_PpIX, emitted_singlet_oxygen_Ppp, emitted_singlet_oxygen_total]
