import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import string
import yaml

EXPERIMENT = "635nm10mW_01"
FILENAME_HEADER_LENGHT = 0 # could be 9

# read setup 
with open(f"experiment/{EXPERIMENT}/experimental_setup.yaml") as file:
    experimental_setup = yaml.safe_load(file)

# store setup information
TITLE = experimental_setup['MS_title']
DISPLAYED_DATA = experimental_setup['displayed_data']
MEASURED_RANGE = experimental_setup['measured_range']

def prepare_data(data_folder : string):
    data_files = os.listdir(f'experiment\{data_folder}\data_MS')

    mass = np.arange(50, 1700, 0.0001)
    df = pd.DataFrame(mass)
    df.columns = ["mass"]

    for filename in data_files:
        data = f"experiment/{EXPERIMENT}/data_MS/{filename}"
        
        zeros = np.zeros(len(mass))
        
        value =  pd.read_csv(data, delimiter='\t')
        value = value.astype(float)

        value.columns = ["rounded_mass", filename[FILENAME_HEADER_LENGHT:-4]]

        for i in range(len(value["rounded_mass"])):
            index = np.round((value["rounded_mass"][i] - 50.0) * 10000)
            zeros[int(index)]=value[filename[FILENAME_HEADER_LENGHT:-4]][i]

        new_values = pd.DataFrame(zeros)
        new_values.columns=[filename[FILENAME_HEADER_LENGHT:-4]]

        df = pd.concat([df, new_values], axis =1, join = 'inner')
    print (df)
    return df

def plot_element(df : pd.DataFrame, element_name : string, element_mass : float, interval_lenght : float = 0.5, lines : bool = False):
    upper_thresh = element_mass + interval_lenght/2 
    lower_thresh = element_mass - interval_lenght/2

    up = np.argwhere(np.array(df["mass"])>upper_thresh).min()
    low = np.argwhere(np.array(df["mass"])>lower_thresh).min()
    
    interval_df = df.iloc[low:up, :]
    
    """bar plot"""
    ax = interval_df.plot(x="mass", y=DISPLAYED_DATA, grid = True, alpha=0.5, figsize=(10, 5))
    if (lines):
        ax.axvline(x = 563.658, color='gray', linestyle='--', label = 'PpIX', alpha=0.3), 
        ax.axvline(x = 565.631, color='gray', linestyle='--', label = 'formyl', alpha=0.4)
        ax.axvline(x = 567.603, color='gray', linestyle='--', label = 'diformyl', alpha=0.5)
        ax.axvline(x = 595.657, color='gray', linestyle='--', label = 'Ppp', alpha=0.6)
        
    
    ax.set_title(f"{element_name}, mass: {element_mass}")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
   
    plt.savefig(f'experiment/{EXPERIMENT}/plots/MS_{TITLE}_{element_name}_bar.png', bbox_inches="tight")
    
    """scatter plot"""
    interval_df_Nan = interval_df.replace(to_replace = 0, value = np.nan, inplace=False)

    """
    for column in interval_df_Nan:
        print (column)
        interval_df_Nan[column] = interval_df_Nan[column] - interval_df_Nan["Solvent"]
    """

    ax = interval_df_Nan.plot(x="mass", y = DISPLAYED_DATA, grid = True, linestyle='None', marker='o', markersize=1, alpha=1, figsize=(10, 5))
    
    if (lines):
        ax.axvline(x = 563.658, color='gray', linestyle='--', label = 'PpIX', alpha=0.3), 
        ax.axvline(x = 565.631, color='gray', linestyle='--', label = 'formyl', alpha=0.4)
        ax.axvline(x = 567.603, color='gray', linestyle='--', label = 'diformyl', alpha=0.5)
        ax.axvline(x = 595.657, color='gray', linestyle='--', label = 'Ppp', alpha=0.6)

    ax.set_title(f"{element_name}, mass: {element_mass}")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

    plt.savefig(f'experiment/{EXPERIMENT}/plots/MS_{TITLE}_{element_name}_point.png', bbox_inches="tight")
   
def calculate_decay(df : pd.DataFrame, element_name : string, element_mass : float, interval_lenght : float = 0.05):  
    upper_thresh = element_mass + interval_lenght/2 
    lower_thresh = element_mass - interval_lenght/2

    up = np.argwhere(np.array(df["mass"])>upper_thresh).min()
    low = np.argwhere(np.array(df["mass"])>lower_thresh).min()
    
    interval_df = df.iloc[low:up, :]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(interval_df["mass"], interval_df[DISPLAYED_DATA]) #, grid = True, alpha=0.5, figsize=(10, 5))
    ax1.set_title(f"Evaluted data (interval width: {interval_lenght})")
    ax1.grid()

    mean_df = interval_df.replace(to_replace = 0, value = np.nan, inplace=False)[DISPLAYED_DATA].mean()
    values = (mean_df.values[:-1] - mean_df.values[-1]).tolist()
    values = [pow(10, -10) if x != x else x for x in values]

    # exponential fit: 
    fit = np.polyfit(MEASURED_RANGE, np.log(values), 1, w=np.sqrt(values))

    a = np.exp(fit[1])
    b = fit[0]

    x_fit = np.linspace(0, MEASURED_RANGE[-1], 100)
    y_fit = a * np.exp(b*x_fit)

    # plot measured data points as well as the exponential fit
    ax2.scatter(MEASURED_RANGE, values)
    ax2.plot(x_fit, y_fit, '--k')

    ax2.set_title(f"Mean value and exponential fit\n(a*exp(bt) with a = {a:.2f}, b = {b:.2f})")
    ax2.grid()

    fig.savefig(f'{EXPERIMENT}/plots/MS_{TITLE}_{element_name}_decay.png', bbox_inches="tight")


print ("Preparing data ...")
df = prepare_data(EXPERIMENT)

print ("Plotting data...")

plot_element(df, element_name="all_w_lines", element_mass = (1000 - 50) / 2, interval_lenght= (1000 - 50) / 2, lines=True)

"""detect PpIX at 562.658 (+1 for hydrogen)"""
plot_element(df, element_name="PpIX", element_mass = 563.658) #563.658)
plt.show()

"""detect Ppp at 594.657 (+1 for hydrogen)"""
plot_element(df, element_name="Ppp", element_mass = 595.657) #595.657)
plt.show()

print ("Done.")



 