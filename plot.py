import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams.update({'font.size': 15})
import numpy as np
from concentration_equations import ground_state_concentration
from concentration_equations import reactive_singlet_oxygen
from config import experimental_setup

energy_dose =  np.linspace(0, 100, 1001)
energy_dose_ro =  np.linspace(0, 100, 100001)

# read data from the csv files
concentration_data_10mW = pd.read_csv('10mW.csv')
concentration_data_100mW = pd.read_csv('100mW.csv')

# specify the optimizated value of mu 
mu_PpIX100 = 4.3e-05
mu_Ppp100= 0.00071
mu_PpIX10 = 3.1e-05
mu_Ppp10= 0.000166

def plot_layout():
    # add ticks
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_major_locator(ticker.AutoLocator()) 
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax.yaxis.set_major_locator(ticker.AutoLocator()) 
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.legend(frameon=False, fontsize=12, loc='upper right')

experimental_setup.set_power_density(10)
experimental_setup.set_wavelength(635)

gs_10mW = ground_state_concentration(energy_dose/10*1000,  mu_PpIX10, mu_Ppp10)
plt.plot(concentration_data_10mW["tag"], concentration_data_10mW["PpIX_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{PpIX}\mathrm{(experimental)}$')
plt.plot(concentration_data_10mW["tag"], concentration_data_10mW["Ppp_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{Ppp}\mathrm{(experimental)}$')

plt.plot(energy_dose, gs_10mW[0], '--', color='blue', label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')
plt.plot(energy_dose, gs_10mW[1], '--', color='orange', label=r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')

plt.ylabel( r'Concentration [$\mu$M]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
plot_layout()
plt.savefig("plots/10mW_gs.png", format='png')
plt.show()

ro_10mW = reactive_singlet_oxygen(energy_dose_ro/10*1000, mu_PpIX10, mu_Ppp10)
print (ro_10mW[2][-1])
plt.plot(energy_dose_ro, ro_10mW[0]/1000, '--', color='blue',  label=r'$[^1O_2]_\mathrm{reactive, PpIX}$')
plt.plot(energy_dose_ro, ro_10mW[1]/1000, '--', color='orange',label=r'$[^1O_2]_\mathrm{reactive, Ppp}$')
plt.plot(energy_dose_ro, ro_10mW[2]/1000, '--', color='green',label=r'$[^1O_2]_\mathrm{reactive, total}$')


plt.ylabel( r'Concentration [mM]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
plot_layout()
#plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
plt.ylim(0, 6.5)
plt.xlim(0, 100)
plot_layout()
plt.legend(frameon=False, fontsize=12, loc='upper left')
plt.savefig("plots/10mW_ro.png", format='png')
plt.show()



experimental_setup.set_power_density(100)

gs_100mW = ground_state_concentration(energy_dose/100*1000, mu_PpIX100, mu_Ppp100)

plt.plot(concentration_data_100mW["tag"], concentration_data_100mW["PpIX_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{PpIX}\mathrm{(experimental)}$')
plt.plot(concentration_data_100mW["tag"], concentration_data_100mW["Ppp_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{Ppp}\mathrm{(experimental)}$')

plt.plot(energy_dose, gs_100mW[0], '--', color='blue', label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')
plt.plot(energy_dose, gs_100mW[1], '--', color='orange', label=r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')

plt.ylabel( r'Concentration [$\mu$M]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
plot_layout()
plt.savefig("plots/100mW_gs.png", format='png')
plt.show()

ro_100mW = reactive_singlet_oxygen(energy_dose_ro/100*1000, mu_PpIX100, mu_Ppp100)
print (ro_100mW[2][-1])
plt.plot(energy_dose_ro, ro_100mW[0]/1000, '--', color='blue',  label=r'$[^1O_2]_\mathrm{reactive, PpIX}$')
plt.plot(energy_dose_ro, ro_100mW[1]/1000, '--', color='orange',label=r'$[^1O_2]_\mathrm{reactive, Ppp}$')
plt.plot(energy_dose_ro, ro_100mW[2]/1000, '--', color='green',label=r'$[^1O_2]_\mathrm{reactive, total}$')

plt.ylabel( r'Concentration [mM]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
plt.legend(frameon=False)
plot_layout()
plt.ylim(0, 6.5)
plt.xlim(0, 100)
plt.legend(frameon=False, fontsize=12, loc='upper left')
#plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
plt.savefig("plots/100mW_ro.png", format='png')
plt.show()


import numpy as np
from scipy.optimize import curve_fit

x_10mW = concentration_data_10mW["tag"]
y_10mW = concentration_data_10mW["PpIX_value"]

x_100mW = concentration_data_100mW["tag"]
y_100mW = concentration_data_100mW["PpIX_value"]


# Kombiniere beide Datensätze
x = np.concatenate([x_10mW, x_100mW])
y = np.concatenate([y_10mW, y_100mW])

def func(x, a):
    return 10*np.exp(a*x)

# Curve Fitting
popt, pcov = curve_fit(func, x, y)

# Angepasste Parameter
a_opt= popt
print (f'aopt: {a_opt/0.0282}')
exp_decay = func(energy_dose, a_opt)

from sklearn.metrics import mean_squared_error

mse_10mW_exp = mean_squared_error(y_10mW, func(x_10mW, a_opt))
mse_100mW_exp = mean_squared_error(y_100mW, func(x_100mW, a_opt))

plt.plot(concentration_data_10mW["tag"], concentration_data_10mW["PpIX_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{PpIX}\mathrm{(experimental)}$')
plt.plot(energy_dose, gs_10mW[0], '--', color='blue', label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')
plt.plot(energy_dose, exp_decay, '--', color='red',  label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(exp. decay model)')


plt.ylabel( r'Concentration [$\mu$M]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
#plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
plot_layout()
plt.xlim(0,100)
plt.ylim(0, 10)
plt.savefig("plots/exp_decay10.png", format='png')
plt.show()

plt.plot(concentration_data_100mW["tag"], concentration_data_100mW["PpIX_value"], "x", label=r'$[S_\mathrm{0}]_\mathrm{PpIX}\mathrm{(experimental)}$')
plt.plot(energy_dose, gs_100mW[0], '--', color='blue', label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(proposed model)')
plt.plot(energy_dose, exp_decay, '--', color='red',  label = r'$[S_\mathrm{0}]_\mathrm{PpIX}$'+'(exp. decay model)')

plt.ylabel( r'Concentration [$\mu$M]')
plt.xlabel( r'Fluence [$\mathrm{J/cm}^2$]')
plot_layout()
plt.xlim(0,100)
plt.ylim(0, 10)
#plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
plt.savefig("plots/exp_decay100.png", format='png')
plt.show()

mse_10mW_new = mean_squared_error(y_10mW, gs_10mW[0][np.linspace(0, 1000, 11).astype(int)])
mse_100mW_new = mean_squared_error(y_100mW, gs_100mW[0][np.linspace(0, 1000, 11).astype(int)])

print ("MSE:")
print(f'new model: 10mW: {mse_10mW_new:.4f}, 100mW: {mse_100mW_new:.4f}, total {(mse_10mW_new+mse_100mW_new/2):.4f}')
print(f'exponential model: 10mW: {mse_10mW_exp:.4f}, 100mW: {mse_100mW_exp:.4f}, total {(mse_10mW_exp+mse_100mW_exp/2):.4f}')
