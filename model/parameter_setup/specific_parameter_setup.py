class SpecificParameterSetup:   
    def __init__(self,
                 xi_PpIX, xi_Ppp, beta_PpIX, beta_Ppp, mu_PpIX, mu_Ppp, delta_PpIX, delta_Ppp,
                 Phi_t_PpIX, Phi_t_Ppp, S_Delta_PpIX, S_Delta_Ppp, gamma_PpIX, gamma_Ppp):
        """ 
        This object manages all experimental independent parameters that are needed for the mathematical model of PDT. 
        For explanations about the meaning of the parameters please refer to the literature or the explanation inside hthe config.py file.

        Args:
            xi_PpIX (float): Parameter xi for PpIX.
            xi_Ppp (float): Parameter xi for Ppp.
            beta_PpIX (float): Parameter beta for PpIX.
            beta_Ppp (float): Parameter beta for Ppp.
            mu_PpIX (float): Parameter mu for PpIX.
            mu_Ppp (float): Parameter mu for Ppp.
            delta_PpIX (float): Parameter delta for PpIX.
            delta_Ppp (float): Parameter delta for Ppp.
            Phi_t_PpIX (float): Parameter Phi_t for PpIX.
            Phi_t_Ppp (float): Parameter Phi_t for Ppp.
            S_Delta_PpIX (float): Parameter S_Delta for PpIX.
            S_Delta_Ppp (float): Parameter S_Delta for Ppp.
            gamma_PpIX (float): Parameter gamma for PpIX
            gamma_Ppp (float): Parameter gamma for Ppp
        """
        # Initialize all the parameters
        self.xi_PpIX = xi_PpIX
        self.xi_Ppp = xi_Ppp
        self.beta_PpIX = beta_PpIX
        self.beta_Ppp = beta_Ppp
        self.mu_PpIX = mu_PpIX
        self.mu_Ppp = mu_Ppp
        self.delta_PpIX = delta_PpIX
        self.delta_Ppp = delta_Ppp
        self.Phi_t_PpIX = Phi_t_PpIX
        self.Phi_t_Ppp = Phi_t_Ppp
        self.S_Delta_PpIX = S_Delta_PpIX
        self.S_Delta_Ppp = S_Delta_Ppp
        self.gamma_PpIX = gamma_PpIX
        self.gamma_Ppp = gamma_Ppp

    # Getter methods for accessing the individual parameter values
    def get_xi_PpIX(self):
        return self.xi_PpIX

    def get_xi_Ppp(self):
        return self.xi_Ppp

    def get_beta_PpIX(self):
        return self.beta_PpIX

    def get_beta_Ppp(self):
        return self.beta_Ppp

    def get_mu_PpIX(self):
        return self.mu_PpIX

    def get_mu_Ppp(self):
        return self.mu_Ppp

    def get_delta_PpIX(self):
        return self.delta_PpIX

    def get_delta_Ppp(self):
        return self.delta_Ppp

    def get_Phi_t_PpIX(self):
        return self.Phi_t_PpIX

    def get_Phi_t_Ppp(self):
        return self.Phi_t_Ppp

    def get_mu_PpIX(self):
        return self.mu_PpIX

    def get_mu_Ppp(self):
        return self.mu_Ppp

    def get_S_Delta_PpIX(self):
        return self.S_Delta_PpIX

    def get_S_Delta_Ppp(self):
        return self.S_Delta_Ppp
    
    def get_gamma_PpIX(self):
        return self.gamma_PpIX

    def get_gamma_Ppp(self):
        return self.gamma_Ppp

    # Method to return all parameters at once
    def get_all_parameters(self):
        """
        Returns a list of all parameters.

        Returns:
            List: A list containing all the parameters.
        """
        return [
            self.xi_PpIX, self.xi_Ppp, self.beta_PpIX, self.beta_Ppp, self.mu_PpIX, self.mu_Ppp,
            self.delta_PpIX, self.delta_Ppp, self.Phi_t_PpIX, self.Phi_t_Ppp, self.S_Delta_PpIX, self.S_Delta_Ppp, self.gamma_PpIX, self.gamma_Ppp
        ]

    # Setter methods for modifying individual parameter values
    def set_xi_PpIX(self, xi_PpIX):
        self.xi_PpIX = xi_PpIX

    def set_xi_Ppp(self, xi_Ppp):
        self.xi_Ppp = xi_Ppp

    def set_beta_PpIX(self, beta_PpIX):
        self.beta_PpIX = beta_PpIX

    def set_beta_Ppp(self, beta_Ppp):
        self.beta_Ppp = beta_Ppp

    def set_mu_PpIX(self, mu_PpIX):
        self.mu_PpIX = mu_PpIX

    def set_mu_Ppp(self, mu_Ppp):
        self.mu_Ppp = mu_Ppp

    def set_delta_PpIX(self, delta_PpIX):
        self.delta_PpIX = delta_PpIX

    def set_delta_Ppp(self, delta_Ppp):
        self.delta_Ppp = delta_Ppp

    def set_Phi_t_PpIX(self, Phi_t_PpIX):
        self.Phi_t_PpIX = Phi_t_PpIX

    def set_Phi_t_Ppp(self, Phi_t_Ppp):
        self.Phi_t_Ppp = Phi_t_Ppp

    def set_mu_PpIX(self, mu_PpIX):
        self.mu_PpIX = mu_PpIX

    def set_mu_Ppp(self, mu_Ppp):
        self.mu_Ppp = mu_Ppp

    def set_S_Delta_PpIX(self, S_Delta_PpIX):
        self.S_Delta_PpIX = S_Delta_PpIX

    def set_S_Delta_Ppp(self, S_Delta_Ppp):
        self.S_Delta_Ppp = S_Delta_Ppp

    def set_gamma_PpIX(self, gamma_PpIX):
        self.gamma_PpIX = gamma_PpIX

    def set_gamma_Ppp(self, gamma_Ppp):
        self.gamma_Ppp = gamma_Ppp

    def set_all_parameters(self, parameters):
        """
        Sets values for all parameters using a list of values.

        Args:
            parameters (List): A list of values corresponding to all parameters.
        """
        (
            self.xi_PpIX, self.xi_Ppp, self.beta_PpIX, self.beta_Ppp, self.mu_PpIX, self.mu_Ppp,
            self.delta_PpIX, self.delta_Ppp, self.Phi_t_PpIX, self.Phi_t_Ppp,
            self.mu_PpIX, self.mu_Ppp, self.S_Delta_PpIX, self.S_Delta_Ppp, self.gamma_PpIX, self.gamma_Ppp
        ) = parameters
