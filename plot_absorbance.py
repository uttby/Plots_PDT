import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
plt.rcParams.update({'font.size': 15})
from scipy import constants as sc

from config import experimental_setup
from config import concentration_PpIX_100J, concentration_Ppp_100J
print (experimental_setup.wavelength)
print(experimental_setup.power_density)
experimental_setup.set_power_density(100)
experimental_setup.set_wavelength(635)
print ("PpIX")
print(experimental_setup.get_APR_PpIX())
print ("Ppp")
print(experimental_setup.get_APR_Ppp())
experimental_setup.set_power_density(10)
experimental_setup.set_wavelength(635)
print ("PpIX")
print(experimental_setup.get_APR_PpIX())
print ("Ppp")
print(experimental_setup.get_APR_Ppp())

# read data
datafile = "data_files/absorbance_PpIX_635nm_10mW_0J_100J.csv"
data =  pd.read_csv(datafile, delimiter=',')


plt.plot(data["wavelenght"][250:], data["0J"][250:], "-", label="before irradiation")
plt.plot(data["wavelenght"][250:], data["100J"][250:], "-", label="after irradiation")

plt.xlabel( 'Wavelength [nm]')
plt.ylabel( 'Absorbance [-]')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

# add ticks
ax = plt.gca()  # Get the current axis
ax.xaxis.set_major_locator(ticker.AutoLocator()) 
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.yaxis.set_major_locator(ticker.AutoLocator()) 
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(which='major', width=1.00, length=5)
ax.tick_params(which='minor', width=0.75, length=2.5)

plt.xlim(350,750)
plt.ylim(0, 1.5)

plt.xticks(np.arange(350, 750+1, 50.0))
plt.legend(frameon=False, fontsize=12, loc='upper right')
plt.savefig("plots/absorbance_PpIX_0J.png", format='png')
plt.show()

plt.plot((concentration_PpIX_100J*experimental_setup.epsilon_PpIX[250:]), "-", label="PpIX")
plt.plot((concentration_Ppp_100J*experimental_setup.epsilon_Ppp[250:]), "-", label="Ppp")


plt.xlabel( 'Wavelength [nm]')
plt.ylabel( 'Absorbance [-]')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

# add ticks
ax = plt.gca()  # Get the current axis
ax.xaxis.set_major_locator(ticker.AutoLocator()) 
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.yaxis.set_major_locator(ticker.AutoLocator()) 
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(which='major', width=1.00, length=5)
ax.tick_params(which='minor', width=0.75, length=2.5)
plt.xlim(350,750)
plt.ylim(0, 0.5)

plt.xticks(np.arange(350, 750+1, 50.0))

plt.legend(frameon=False, fontsize=12, loc='upper right')
plt.savefig("plots/absorbance_100J_all.png", format='png')
plt.show()


plt.plot(experimental_setup.epsilon_PpIX * (2.3025 * 1000 / sc.Avogadro), "-", label="PpIX")
plt.plot(experimental_setup.epsilon_Ppp * (2.3025 * 1000 / sc.Avogadro), "-", label="Ppp")

plt.xlabel( 'Wavelength [nm]')
plt.ylabel( 'Absorption Cross Section '+r'$\mathrm{[cm}^2]$')
plt.legend(frameon=False, fontsize=12, loc='upper right')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)


# add ticks
ax = plt.gca()  # Get the current axis
ax.xaxis.set_major_locator(ticker.AutoLocator()) 
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.yaxis.set_major_locator(ticker.AutoLocator()) 
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()) 
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(which='major', width=1.00, length=5)
ax.tick_params(which='minor', width=0.75, length=2.5)

plt.xlim(350,750)
plt.ylim(0,2e-15)
plt.savefig("plots/absorbance_cross_section.png", format='png')
plt.show()

print (experimental_setup.epsilon_PpIX)
