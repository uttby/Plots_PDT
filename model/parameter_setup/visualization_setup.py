# Stores information about the visualization of the plots, for example wether they should be plotted over time of power dose
class VisualizationSetup():
    def __init__(self, x_label, x_factor, additional_data_traces = None):
        """ 
        This object manages the visualization properties of the plots, mostly regarding whether the 
        x axis is supposed to show the energy dose or time. 
        It manages the x label, factor, but also additional data traces shown in the ground state concentration plot and the optimization results used to generate
        optimization history plots.

        Args:
            x_label (String): Label for x axis, normally either 'Time' or 'Energy Dose'
            x_factor (float): Set this to 1, if the time should be plotted (t*1 = t). If the energy dose is plotted, this factor should equal to the power density of the irradiation laser 
            in Watt/cm^2 (t*power_dose = energy_dose)
            additional_data_traces (plotly.graph_objects.go object): Scatter plot including the newly uploaded mass spectrometry data
        """
        # Initialize all the parameters
        self.x_label = x_label
        self.x_factor = x_factor
        self.additional_data_traces = additional_data_traces
    
    def get_x_label(self):
        return self.x_label
    
    def get_x_factor(self):
        return self.x_factor
    
    def get_additional_data_traces(self):
        return self.additional_data_traces
    
    def set_x_label(self, x_label):
        self.x_label = x_label
    
    def set_x_factor(self, x_factor):
        self.x_factor = x_factor

    def set_additional_data_traces(self, additional_data_traces):
        self.additional_data_traces = additional_data_traces

    def add_additional_data_traces(self, additional_data_trace):
        self.additional_data_traces.add(additional_data_trace)

    def set_opt_results(self, results):
        self.opt_results = results

    def get_opt_results(self):
        return(self.opt_results)