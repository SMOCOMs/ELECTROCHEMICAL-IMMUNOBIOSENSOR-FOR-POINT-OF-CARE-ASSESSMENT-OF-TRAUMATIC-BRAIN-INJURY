#Final vertion to automate the PSsoftwar data extration and/or the scientific plot of the data


#This file can be adaptated to your specific type of data or computer as well as the nomenclature

# The ecxel files are from PSsoftware. when with _fit means that is from the fiting PSsoftware.

# The .txt file I use in my reserache is:
# Rs,ERs,Rct,ERct
# valueRs,value%ERs,valueRct,value%ERct
 

# Well when we are working with EIS measurments the most important plot is the nyquist plot and the 
# relation with each sample. So the first def will be, in this case, to understande the funcionalization 
# of electrod's surface with a bar plot. This can be extended to the measumrnet of the analyte (GFAP)

def Rct_Bio_Funcionalization_Steps_plot_all(Directory,finalDirectory,file_names,Probe,probemolecule,BSA,combination):
    """
    Plots all the Rct values from multiple files on the same figure.

    Parameters:
    - Directory: str, the directory where files are stored.
    - finalDirectory : str, the directory where the final file will be stored.
    - file_names: list of str, the base names of the files.
    - Probe: str, the probe type used in the experiment.
    - probemolecule: str, the name of the probe molecule.
    - BSA: str, the name of the BSA molecule or file suffix.
    - combination: str, the combination of data files to process.
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    if combination == 'gfap':
        output_name1= 'GFAP pg/mL'
        output_name2= 'GFAP'
    else:
        output_name1= 'Functionalization Steps'
        output_name2= 'Bio_Funcionalization_Steps'

    fig, ax = plt.subplots(figsize=(15, 7))
    
    for colornumber, file_name in enumerate(file_names, start=1):
        Reading_Rct_Bio_Funcionalization_Steps(Directory, file_name, Probe, ax, colornumber, probemolecule, BSA, combination)
    
    # Set the output directory where the plot will be saved
    os.chdir(finalDirectory)
    
    ax.set_ylabel("Rct / Ohm", fontsize=20, fontweight='bold')
    ax.set_xlabel(f"{output_name1}", fontsize=20, fontweight='bold')
    ax.set_title(f"Electrochemical Impedance Spectroscopy - {', '.join(file_names)}", fontweight='bold')
    ax.legend(loc=1, bbox_to_anchor=(1, 1), fontsize='x-large')
    ax.tick_params(axis='both', labelsize=16)
    
    output_file = f"Electrochemical_Impedance_Spectroscopy(Rct)_barplot_{'_'.join(file_names)}_{output_name2}.png"
    plt.savefig(output_file)
    plt.show()

def Reading_Rct_Bio_Funcionalization_Steps(Directory,file_name,Probe,ax,colornumber,probemolecule,BSA,combination):
    '''
    Reads various data files, extracts Rct values and their errors,
    and plots these values using a bar chart with error bars.
    
    Parameters:
    - Directory: str, the directory where files are stored.
    - file_name: str, the base name of the files.
    - Probe: str, the probe type used in the experiment.
    - ax: matplotlib axis, the axis to plot the bar chart on.
    - colornumber: int, the index for selecting the bar color.
    - probemolecule: str, the name of the probe molecule.
    - BSA: str, the name of the BSA molecule or file suffix.
    - combination: str, the combination of data files to process.
    '''
    
    import matplotlib.pyplot as plt
    import os
    import matplotlib.colors as mcolors
    import pandas as pd

    colors = list(mcolors.XKCD_COLORS.values())

    os.chdir(Directory)
    
    file_patterns = {
        'NC': f'{file_name}_Not_Clean_{Probe}.txt',
        'C': f'{file_name}_Clean_{Probe}.txt',
        'MBA': f'{file_name}_{probemolecule}_{Probe}.txt',
        'Ab': f'{file_name}_Ab_{Probe}.txt',
        'BSA': f'{file_name}_{BSA}_{Probe}.txt',
        'pg1_15': f'{file_name}_1_15pg_{Probe}.txt',
        'pg1_54': f'{file_name}_1_54pg_{Probe}.txt',
        'pg2_06': f'{file_name}_2_06pg_{Probe}.txt',
        'pg6_17': f'{file_name}_6_17pg_{Probe}.txt',
        'pg18_53': f'{file_name}_18_53pg_{Probe}.txt',
        'pg55_6': f'{file_name}_55_6pg_{Probe}.txt',
        'pg167': f'{file_name}_0_167ng_{Probe}.txt',
        'pg500': f'{file_name}_0_5ng_{Probe}.txt',
    }

    combinations = {
        'all': ['NC', 'C', 'MBA', 'Ab', 'BSA'],
        'NC,C,MBA,Ab': ['NC', 'C', 'MBA', 'Ab'],
        'NC,C,Ab,BSA':['NC', 'C', 'Ab', 'BSA'],
        'NC,C,MBA': ['NC', 'C', 'MBA'],
        'NC,MBA,Ab,BSA': ['NC', 'MBA', 'Ab', 'BSA'],
        'MBA,Ab,BSA': ['MBA', 'Ab', 'BSA'],
        'NC,C,MBA,BSA': ['NC', 'C', 'MBA', 'BSA'],
        'gfap': ['pg1_15', 'pg1_54', 'pg2_06', 'pg6_17', 'pg18_53', 'pg55_6', 'pg167', 'pg500']
    }

    selected_files = combinations.get(combination, [])
    if not selected_files:
        raise ValueError(f"Invalid combination: {combination}")

    Rct_values = []
    yerr = []
    Rct_names = []

    for key in selected_files:
            data = pd.read_csv(file_patterns[key])
            Rct_values.append(data['Rct'][0])
            yerr.append((data['ERct'][0]*data['Rct'][0])/100)

            # Append corresponding names based on the key
            if key == 'NC':
                Rct_names.append('Not Clean WE')
            elif key == 'C':
                Rct_names.append('Clean WE')
            elif key == 'MBA':
                Rct_names.append(f'{probemolecule}')
            elif key == 'Ab':
                Rct_names.append('Antibody')
            elif key == 'BSA':
                Rct_names.append(f'{BSA}')
            elif key == 'pg1_15':
                Rct_names.append('1.15 pg')
            elif key == 'pg1_54':
                Rct_names.append('1.54 pg')
            elif key == 'pg2_06':
                Rct_names.append('2.06 pg')
            elif key == 'pg6_17':
                Rct_names.append('6.17 pg')
            elif key == 'pg18_53':
                Rct_names.append('18.53 pg')
            elif key == 'pg55_6':
                Rct_names.append('55.6 pg')
            elif key == 'pg167':
                Rct_names.append('0.167 ng')
            elif key == 'pg500':
                Rct_names.append('0.5 ng')
    
    ax.bar(Rct_names, Rct_values, color='grey')
    ax.errorbar(Rct_names, Rct_values, yerr=yerr, fmt=".", color='black')
    print(Rct_names)
    print(Rct_values)
    print(yerr)

# Scattering a nyquist plot with the relevante surface funtionalisation data

def Nyquist_plot(Directory, file_name, Probe, probemolecule, BSA, combination):
    '''
    This function generates a Nyquist plot based on EIS data stored in Excel files for various biofunctionalization steps.

    Parameters:
    - Directory: str, the directory where files are stored.
    - file_name: str, the base name of the files.
    - Probe: str, the probe type used in the experiment.
    - probemolecule: str, the name of the probe molecule.
    - BSA: str, the name of the blocking molecule.
    - combination: str, the combination of data files to process.
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    os.chdir(Directory)

    # Define file paths for different steps
    file_patterns = {
        'NC': f'{file_name}_Not_Clean_{Probe}.xlsx',
        'C': f'{file_name}_Clean_{Probe}.xlsx',
        'Au': f'{file_name}_Au_{Probe}.xlsx',
        'MBA': f'{file_name}_{probemolecule}_{Probe}.xlsx',
        'Ab': f'{file_name}_Ab_{Probe}.xlsx',
        'BSA': f'{file_name}_{BSA}_{Probe}.xlsx'
    }

    # Define combinations
    combinations = {
        'all': ['NC', 'C', 'Au', 'MBA', 'Ab', 'BSA'],
        'NC,C,MBA,Ab': ['NC', 'C', 'MBA', 'Ab'],
        'NC,C,Au': ['NC', 'C','Au'],
        'NC,C,MBA': ['NC', 'C', 'MBA'],
        'NC,MBA,Ab,BSA': ['NC', 'MBA', 'Ab', 'BSA'],
        'MBA,Ab,BSA': ['MBA', 'Ab', 'BSA'],
        'NC,C,Ab': ['NC', 'C', 'Ab'],
        'NC,C,MBA,BSA': ['NC', 'C', 'MBA', 'BSA']
    }

    # Get the selected files based on the combination
    selected_keys = combinations.get(combination, [])
    if not selected_keys:
        raise ValueError(f"Invalid combination: {combination}")

    # Read and store data frames for selected files
    data_frames = {}
    for key in selected_keys:
        file_path = file_patterns[key]
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df_header = df.iloc[0] # Use the first row as the header
            df = df[1:] # Skip the header row for data
            df.columns = df_header
            data_frames[key] = df    
            
        else:
            print(f"File {file_path} not found.")
    print(data_frames)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot data for each step
    colors = {
        'NC': '#000000',
        'C': '#00FFFF',
        'Au': '#FFD700',
        'MBA': '#FF00FF',
        'Ab': '#FF0000',
        'BSA': '#00FF00'
    }
    labels = {
        'NC': 'Not Clean Surface',
        'C': 'Clean Surface with Sulfuric Acid',
        'Au': 'Surface modification with Gold',
        'MBA': 'Surface conjugation with probe molecules',
        'Ab': 'Surface funtionalization with Anti-GFAP',
        'BSA': 'Surface bloking with BSA'
    }

    for key, df in data_frames.items():
        ax.scatter(df["Z' / Ohm"], df["Z'' / Ohm"], label=labels[key], color=colors[key])

    plt.xlabel("Z' / Ohm", fontsize=12, fontweight='bold')
    plt.ylabel("-Z'' / Ohm", fontsize=12, fontweight='bold')
    plt.title(f"Electrochemical Impedance Spectroscopy Modification Steps - {file_name}", fontweight='bold')

    plt.legend(prop={'size': 8.5, 'weight': 'bold'})

    plt.savefig(f"Electrochemical_Impedance_Spectroscopy_Modification_{file_name}_{Probe}.png")
    plt.show()


# With tne Nyquist plots we can understand minimaly the way how the electrod is reponding however it's to difficult 
# visualize the Rct (Transference Charge resistence) and if it's increasing or not, so to turn this possible and plesurease
# to the reader we need specifically the bar plot Rct vs Concntration or funtionalization step. It's also needded one scatter
# plot to be possible to determine the caracteristic funtion.

def Rct_GFAP_plot_all(Directory,finalDirectory, file_names, Probe, probemolecule, BSA, trendline_start, trendline_end,line_trend):
    """
    Plots Rct values against GFAP concentrations for multiple datasets.
    
    Parameters:
        Directory (str): Path to the directory containing the data files.
        finalDirectory (str): the directory where the final file will be stored.
        file_names (list of str): List of base names for the files.
        Probe (str): Probe identifier used in the file names.
        probemolecule (str): Name of the probe molecule.
        BSA (str): Blocking agent identifier used in the file names.
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Initialize color counter
    colornumber = 0
    
    # Loop through each file name and plot its data
    for file_name in file_names:
        colornumber += 1 
        Reading_Rct_GFAP(Directory, file_name, Probe, ax, colornumber, probemolecule, BSA, trendline_start, trendline_end,line_trend)

    # Set the output directory where the plot will be saved
    os.chdir(finalDirectory)

    ax.set_ylabel("Rct / Ohm", fontsize=20, fontweight='bold')
    ax.set_xlabel("GFAP pg/mL", fontsize=20, fontweight='bold')
    ax.set_title(f"Electrochemical Impedance Spectroscopy - {file_names}", fontweight='bold')
    ax.legend(loc=1, bbox_to_anchor=(1, 1), fontsize='x-large')
    ax.tick_params(axis='both', labelsize=16)
    
    plt.savefig(f"Electrochemical_Impedance_Spectroscopy_Rct_plot_{'_'.join(file_names)}_GFAP.png")
    plt.show()
 

def Rct_logGFAP_plot_all(Directory,finalDirectory, file_names, Probe, probemolecule, BSA, trendline_start, trendline_end):
    """
    Plots Rct values for multiple files against log(GFAP) concentrations.

    Parameters:
        Directory (str): Path to the directory containing the data files.
        finalDirectory (str): the directory where the final file will be stored.
        file_names (list): List of file name prefixes.
        Probe (str): Probe identifier used in the file names.
        probemolecule (str): Name of the probe molecule.
        BSA (str): BSA identifier used in the file names.
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    
    fig, bx = plt.subplots(figsize=(15, 7))

    # Initialize color counter
    colornumber = 0     

    # Loop through each file name and plot its data               
    for file_name in file_names:
        colornumber += 1
        Reading_Rct_logGFAP(Directory, file_name, Probe, bx, colornumber, probemolecule, BSA, trendline_start, trendline_end)

    # Set the output directory where the plot will be saved
    os.chdir(finalDirectory)

    bx.set_ylabel("Rct / Ohm", fontsize=20, fontweight='bold')
    bx.set_xlabel("log(GFAP) pg/mL", fontsize=20, fontweight='bold')
    bx.set_title(f"Electrochemical Impedance Spectroscopy - {', '.join(file_names)}", fontweight='bold')
    bx.legend(loc=1, bbox_to_anchor=(1,1), fontsize='x-large')
    bx.tick_params(axis='both', labelsize=16)

    plt.savefig(f"Electrochemical_Impedance_Spectroscopy_Rct_plotlog_{'_'.join(file_names)}_GFAP.png")

    plt.show()

def Reading_Rct_GFAP(Directory, file_name, Probe, ax, colornumber, probemolecule, BSA, trendline_start, trendline_end,line_trend):
    '''
    Reads Rct and ERct values from files, processes them, and plots the data.
    
    Parameters:
        Directory (str): Path to the directory containing the data files.
        file_name (str): Base name for the files.
        Probe (str): Probe identifier used in the file names.
        bx (matplotlib.axes.Axes): Axes object where the data will be plotted.
        colornumber (int): Index for the color to use in the plot.
        probemolecule (str): Name of the probe molecule.
        BSA (str): Bloking agent identifier used in the file names.
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.colors as mcolors
    import os
    import fnmatch

    colors = list(mcolors.XKCD_COLORS.values())
    os.chdir(Directory)

    file_patterns = {
        f'{probemolecule}': f'{file_name}_{probemolecule}_{Probe}.txt',
        'Ab': f'{file_name}_Ab_{Probe}.txt',
        f'{BSA}': f'{file_name}_{BSA}_{Probe}.txt',
        'pg0_04': f'{file_name}_0_04pg_{Probe}.txt',
        'pg0_07': f'{file_name}_0_07pg_{Probe}.txt',
        'pg0_14': f'{file_name}_0_14pg_{Probe}.txt',
        'pg0_29': f'{file_name}_0_29pg_{Probe}.txt',
        'pg0_58': f'{file_name}_0_58pg_{Probe}.txt',
        'pg1_15': f'{file_name}_1_15pg_{Probe}.txt',
        'pg1_54': f'{file_name}_1_54pg_{Probe}.txt',
        'pg2_06': f'{file_name}_2_06pg_{Probe}.txt',
        'pg6_17': f'{file_name}_6_17pg_{Probe}.txt',
        'pg18_53': f'{file_name}_18_53pg_{Probe}.txt',
        'pg55_6': f'{file_name}_55_6pg_{Probe}.txt',
        'pg167': f'{file_name}_0_167ng_{Probe}.txt',
        'pg500': f'{file_name}_0_5ng_{Probe}.txt',
    }

    combination = []
    for key, pattern in file_patterns.items():
        if any(fnmatch.fnmatch(file, pattern) for file in os.listdir(Directory)):
            combination.append(key)

    # Remove redundant entries
    if f'{BSA}' in combination and not 'Ab'  in combination:
        print('only BSA')
    elif f'{BSA}' in combination:
        combination.remove('Ab')
        combination.remove(f'{probemolecule}')
    elif 'Ab' in combination and not f'{probemolecule}' in combination:
        print('only Ab')
    elif 'Ab' in combination:
        combination.remove(f'{probemolecule}')
    


    Rct_values, yerr, Rct_names, GFAP_concentrations = [], [], [], []

    for key in combination:
        data = pd.read_csv(file_patterns[key])
        Rct_values.append(data['Rct'][0])
        yerr.append((data['ERct'][0] * data['Rct'][0]) / 100)

        if key in [f'{probemolecule}', 'Ab', f'{BSA}']:
            Rct_names.append(key)
            GFAP_concentrations.append(0)
        elif key == 'pg0_04':
            Rct_names.append('0.04 pg')
            GFAP_concentrations.append(0.04)
        elif key == 'pg0_07':
            Rct_names.append('0.07 pg')
            GFAP_concentrations.append(0.07)
        elif key == 'pg0_14':
            Rct_names.append('0.14 pg')
            GFAP_concentrations.append(0.14)
        elif key == 'pg0_29':
            Rct_names.append('0.29 pg')
            GFAP_concentrations.append(0.29)
        elif key == 'pg0_58':
            Rct_names.append('0.58 pg')
            GFAP_concentrations.append(0.58)
        elif key == 'pg1_15':
            Rct_names.append('1.15 pg')
            GFAP_concentrations.append(1.15)
        elif key == 'pg1_54':
            Rct_names.append('1.54 pg')
            GFAP_concentrations.append(1.54)
        elif key == 'pg2_06':
            Rct_names.append('2.06 pg')
            GFAP_concentrations.append(2.06)
        elif key == 'pg6_17':
            Rct_names.append('6.17 pg')
            GFAP_concentrations.append(6.17)
        elif key == 'pg18_53':
            Rct_names.append('18.53 pg')
            GFAP_concentrations.append(18.53)
        elif key == 'pg55_6':
            Rct_names.append('55.6 pg')
            GFAP_concentrations.append(55.6)
        elif key == 'pg167':
            Rct_names.append('0.167 ng')
            GFAP_concentrations.append(167)
        elif key == 'pg500':
            Rct_names.append('0.5 ng')
            GFAP_concentrations.append(500)
    
    print(Rct_values)
    
    
    ax.plot(GFAP_concentrations, Rct_values, label=f'{file_name}', color=colors[colornumber])
    ax.scatter(GFAP_concentrations, Rct_values, color='black')
    ax.errorbar(GFAP_concentrations, Rct_values, yerr=yerr, fmt=".", color='black')
    

    if line_trend == 0:
        print('no trend line')

    if line_trend == 1:
        if len(GFAP_concentrations) > 1:
            # Select the concentration and Rct data for trendline fitting
            filtered_concentrations = GFAP_concentrations[trendline_start:trendline_end]
            filtered_Rct_values = Rct_values[trendline_start:trendline_end]
            
        if len(filtered_concentrations) > 1:  # Ensure there's enough data to fit
            coefficients = np.polyfit(filtered_concentrations, filtered_Rct_values, 1)  # Linear fit
            polynomial = np.poly1d(coefficients)
            trendline = polynomial(np.sort(filtered_concentrations))
            ax.plot(np.sort(filtered_concentrations), trendline, linestyle='--', color=colors[colornumber], label=f'{file_name} Trend line')

        # Calculate R^2
        y_pred = polynomial(filtered_concentrations)
        residuals = np.array(filtered_Rct_values) - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(filtered_Rct_values) - np.mean(filtered_Rct_values))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Display trendline equation and R^2
        trendline_eq = f'Rct = {coefficients[0]:.2f} * [GFAP] + {coefficients[1]:.2f}'
        print(f'Trendline equation {file_name}: {trendline_eq}')
        print(f'R^2: {r2:.4f}')

def Reading_Rct_logGFAP(Directory, file_name, Probe, bx, colornumber, probemolecule, BSA, trendline_start, trendline_end):
    '''
    Reads Rct and ERct values from files, processes them, and plots the data.
    
    Parameters:
        Directory (str): Path to the directory containing the data files.
        file_name (str): Base name for the files.
        Probe (str): Probe identifier used in the file names.
        bx (matplotlib.axes.Axes): Axes object where the data will be plotted.
        colornumber (int): Index for the color to use in the plot.
        probemolecule (str): Name of the probe molecule.
        BSA (str): BSA identifier used in the file names.
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.colors as mcolors
    import os
    import fnmatch

    colors = list(mcolors.XKCD_COLORS.values())
    os.chdir(Directory)

    file_patterns = {
        f'{probemolecule}': f'{file_name}_{probemolecule}_{Probe}.txt',
        'Ab': f'{file_name}_Ab_{Probe}.txt',
        f'{BSA}': f'{file_name}_{BSA}_{Probe}.txt',
        'pg0_04': f'{file_name}_0_04pg_{Probe}.txt',
        'pg0_07': f'{file_name}_0_07pg_{Probe}.txt',
        'pg0_14': f'{file_name}_0_14pg_{Probe}.txt',
        'pg0_29': f'{file_name}_0_29pg_{Probe}.txt',
        'pg0_58': f'{file_name}_0_58pg_{Probe}.txt',
        'pg1_15': f'{file_name}_1_15pg_{Probe}.txt',
        'pg1_54': f'{file_name}_1_54pg_{Probe}.txt',
        'pg2_06': f'{file_name}_2_06pg_{Probe}.txt',
        'pg6_17': f'{file_name}_6_17pg_{Probe}.txt',
        'pg18_53': f'{file_name}_18_53pg_{Probe}.txt',
        'pg55_6': f'{file_name}_55_6pg_{Probe}.txt',
        'pg167': f'{file_name}_0_167ng_{Probe}.txt',
        'pg500': f'{file_name}_0_5ng_{Probe}.txt',   
    }

    combination = []
    for key, pattern in file_patterns.items():
        if any(fnmatch.fnmatch(file, pattern) for file in os.listdir(Directory)):
            combination.append(key)

    # Remove redundant entries
    if f'{BSA}' in combination and not 'Ab' in combination:
        print('only BSA')
    elif f'{BSA}' in combination:
        combination.remove('Ab')
        combination.remove(f'{probemolecule}')
    elif 'Ab' in combination and not f'{probemolecule}' in combination:
        print('only Ab')
    elif 'Ab' in combination:
        combination.remove(f'{probemolecule}')
    Rct_values, yerr, Rct_names, GFAP_concentrations = [], [], [], []

    for key in combination:
        data = pd.read_csv(file_patterns[key])
        Rct_values.append(data['Rct'][0])
        yerr.append((data['ERct'][0] * data['Rct'][0]) / 100)

        if key in [f'{probemolecule}', 'Ab', f'{BSA}']:
            Rct_names.append(key)
            GFAP_concentrations.append(0)
        elif key == 'pg0_04':
            Rct_names.append('0.04 pg')
            GFAP_concentrations.append(0.04)
        elif key == 'pg0_07':
            Rct_names.append('0.07 pg')
            GFAP_concentrations.append(0.07)
        elif key == 'pg0_14':
            Rct_names.append('0.14 pg')
            GFAP_concentrations.append(0.14)
        elif key == 'pg0_29':
            Rct_names.append('0.29 pg')
            GFAP_concentrations.append(0.29)
        elif key == 'pg0_58':
            Rct_names.append('0.58 pg')
            GFAP_concentrations.append(0.58)
        elif key == 'pg1_15':
            Rct_names.append('1.15 pg')
            GFAP_concentrations.append(1.15)
        elif key == 'pg1_54':
            Rct_names.append('1.54 pg')
            GFAP_concentrations.append(1.54)
        elif key == 'pg2_06':
            Rct_names.append('2.06 pg')
            GFAP_concentrations.append(2.06)
        elif key == 'pg6_17':
            Rct_names.append('6.17 pg')
            GFAP_concentrations.append(6.17)
        elif key == 'pg18_53':
            Rct_names.append('18.53 pg')
            GFAP_concentrations.append(18.53)
        elif key == 'pg55_6':
            Rct_names.append('55.6 pg')
            GFAP_concentrations.append(55.6)
        elif key == 'pg167':
            Rct_names.append('0.167 ng')
            GFAP_concentrations.append(167)
        elif key == 'pg500':
            Rct_names.append('0.5 ng')
            GFAP_concentrations.append(500)

    log_GFAP_Concentrations = np.log10(GFAP_concentrations)
    log_Rct = np.log10(Rct_values)

    bx.plot(log_GFAP_Concentrations, Rct_values, label=f'{file_name}', color=colors[colornumber])
    bx.scatter(log_GFAP_Concentrations, Rct_values, color='#000000')
    bx.errorbar(log_GFAP_Concentrations, Rct_values, yerr=yerr, fmt=".", color='black')
    
    if trendline_start is None:
        trendline_start = 0
    if trendline_end is None:
        trendline_end = len(log_GFAP_Concentrations)

    # Ensure the indices are within range
    trendline_start = max(0, trendline_start)
    trendline_end = min(len(log_GFAP_Concentrations), trendline_end)

     # Default trendline points to include all if not specified
    if trendline_start is None:
        trendline_start = 0
    if trendline_end is None:
        trendline_end = len(log_GFAP_Concentrations)

    # Ensure the indices are within range
    trendline_start = max(0, trendline_start)
    trendline_end = min(len(log_GFAP_Concentrations), trendline_end)

    if trendline_end - trendline_start > 1:
    # Perform the linear fit on the log-transformed GFAP concentrations
        filtered_log_concentrations = log_GFAP_Concentrations[trendline_start:trendline_end]
        filtered_Rct_values = Rct_values[trendline_start:trendline_end]
    
    # Perform a linear fit
    coefficients = np.polyfit(filtered_log_concentrations, filtered_Rct_values, 1)  # Linear fit
    polynomial = np.poly1d(coefficients)
    trendline = polynomial(np.sort(filtered_log_concentrations))
    
    # Plot the trendline
    bx.plot(np.sort(filtered_log_concentrations), trendline, linestyle='--', color=colors[colornumber], label=f'{file_name} Trend line')

    # Calculate R^2
    y_pred = polynomial(filtered_log_concentrations)
    residuals = np.array(filtered_Rct_values) - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(filtered_Rct_values) - np.mean(filtered_Rct_values))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Display trendline equation and R^2
    trendline_eq = f'Rct = {coefficients[0]:.2f} * log[GFAP] + {coefficients[1]:.2f}'
    print(f'Trendline equation {file_name}: {trendline_eq}')
    print(f'R^2: {r2:.4f}')

def deltaRct_GFAP_plot_all(Directory,finalDirectory, file_names, Probe, probemolecule, BSA, trendline_start, trendline_end,line_trend):
    """
    Plots Rct values against GFAP concentrations for multiple datasets.
    
    Parameters:
        Directory (str): Path to the directory containing the data files.
        finalDirectory (str): the directory where the final file will be stored.
        file_names (list of str): List of base names for the files.
        Probe (str): Probe identifier used in the file names.
        probemolecule (str): Name of the probe molecule.
        BSA (str): Blocking agent identifier used in the file names.
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Initialize color counter
    colornumber = 0
    
    # Loop through each file name and plot its data
    for file_name in file_names:
        colornumber += 1 
        Reading_deltaRct_GFAP(Directory, file_name, Probe, ax, colornumber, probemolecule, BSA, trendline_start, trendline_end,line_trend)

    # Change directory to save the figure
    os.chdir(finalDirectory)

    ax.set_ylabel("ΔRct / Ohm", fontsize=20, fontweight='bold')
    ax.set_xlabel("GFAP pg/mL", fontsize=20, fontweight='bold')
    ax.set_title(f"Electrochemical Impedance Spectroscopy ΔRct - {file_names}", fontweight='bold')
    ax.legend(loc=1, bbox_to_anchor=(1, 1), fontsize='x-large')
    ax.tick_params(axis='both', labelsize=16)
    
    plt.savefig(f"Electrochemical_Impedance_Spectroscopy_deltaRct_plot_{'_'.join(file_names)}_GFAP.png")
    plt.show()

def Reading_deltaRct_GFAP(Directory, file_name, Probe, ax, colornumber, probemolecule, BSA, trendline_start, trendline_end,line_trend):
    '''
    Reads Rct and ERct values from files, processes them, and plots the data.
    
    Parameters:
        Directory (str): Path to the directory containing the data files.
        file_name (str): Base name for the files.
        Probe (str): Probe identifier used in the file names.
        bx (matplotlib.axes.Axes): Axes object where the data will be plotted.
        colornumber (int): Index for the color to use in the plot.
        probemolecule (str): Name of the probe molecule.
        BSA (str): Bloking agent identifier used in the file names.
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.colors as mcolors
    import os
    import fnmatch

    colors = list(mcolors.XKCD_COLORS.values())
    os.chdir(Directory)

    file_patterns = {
        f'{probemolecule}': f'{file_name}_{probemolecule}_{Probe}.txt',
        'Ab': f'{file_name}_Ab_{Probe}.txt',
        f'{BSA}': f'{file_name}_{BSA}_{Probe}.txt',
        'pg0_04': f'{file_name}_0_04pg_{Probe}.txt',
        'pg0_07': f'{file_name}_0_07pg_{Probe}.txt',
        'pg0_14': f'{file_name}_0_14pg_{Probe}.txt',
        'pg0_29': f'{file_name}_0_29pg_{Probe}.txt',
        'pg0_58': f'{file_name}_0_58pg_{Probe}.txt',
        'pg1_15': f'{file_name}_1_15pg_{Probe}.txt',
        'pg1_54': f'{file_name}_1_54pg_{Probe}.txt',
        'pg2_06': f'{file_name}_2_06pg_{Probe}.txt',
        'pg6_17': f'{file_name}_6_17pg_{Probe}.txt',
        'pg18_53': f'{file_name}_18_53pg_{Probe}.txt',
        'pg55_6': f'{file_name}_55_6pg_{Probe}.txt',
        'pg167': f'{file_name}_0_167ng_{Probe}.txt',
        'pg500': f'{file_name}_0_5ng_{Probe}.txt',
    }

    combination = []
    for key, pattern in file_patterns.items():
        if any(fnmatch.fnmatch(file, pattern) for file in os.listdir(Directory)):
            combination.append(key)

    # Remove redundant entries
    if f'{BSA}' in combination and not 'Ab'  in combination:
        print('only BSA')
    elif f'{BSA}' in combination:
        combination.remove('Ab')
        combination.remove(f'{probemolecule}')
    elif 'Ab' in combination and not f'{probemolecule}' in combination:
        print('only Ab')
    elif 'Ab' in combination:
        combination.remove(f'{probemolecule}')

    Rct_values, yerr, Rct_names, GFAP_concentrations, Delta_Rct, Deltayerr = [], [], [], [], [], []

    for key in combination:
        data = pd.read_csv(file_patterns[key])
        Rct_values.append(data['Rct'][0])
        yerr.append((data['ERct'][0] * data['Rct'][0]) / 100)

        if key in [f'{probemolecule}', 'Ab', f'{BSA}']:
            Rct_names.append(key)
            GFAP_concentrations.append(0)
        elif key == 'pg0_04':
            Rct_names.append('0.04 pg')
            GFAP_concentrations.append(0.04)
        elif key == 'pg0_07':
            Rct_names.append('0.07 pg')
            GFAP_concentrations.append(0.07)
        elif key == 'pg0_14':
            Rct_names.append('0.14 pg')
            GFAP_concentrations.append(0.14)
        elif key == 'pg0_29':
            Rct_names.append('0.29 pg')
            GFAP_concentrations.append(0.29)
        elif key == 'pg0_58':
            Rct_names.append('0.58 pg')
            GFAP_concentrations.append(0.58)
        elif key == 'pg1_15':
            Rct_names.append('1.15 pg')
            GFAP_concentrations.append(1.15)
        elif key == 'pg1_54':
            Rct_names.append('1.54 pg')
            GFAP_concentrations.append(1.54)
        elif key == 'pg2_06':
            Rct_names.append('2.06 pg')
            GFAP_concentrations.append(2.06)
        elif key == 'pg6_17':
            Rct_names.append('6.17 pg')
            GFAP_concentrations.append(6.17)
        elif key == 'pg18_53':
            Rct_names.append('18.53 pg')
            GFAP_concentrations.append(18.53)
        elif key == 'pg55_6':
            Rct_names.append('55.6 pg')
            GFAP_concentrations.append(55.6)
        elif key == 'pg167':
            Rct_names.append('0.167 ng')
            GFAP_concentrations.append(167)
        elif key == 'pg500':
            Rct_names.append('0.5 ng')
            GFAP_concentrations.append(500)
    
    print(Rct_values)
    

    if f'{BSA}' in combination or 'Ab' in combination or f'{probemolecule}' in combination:
        for a in range(len(Rct_values)):
            Delta_Rct.append((Rct_values[a]-Rct_values[0])/Rct_values[0])
            Deltayerr.append(calculate_error(yerr[a],yerr[0],Rct_values[a],Rct_values[0]))
    else:
        print("Don't have Rct 0" )
            
    ax.scatter(GFAP_concentrations, Delta_Rct, color='black')
    ax.errorbar(GFAP_concentrations,Delta_Rct, yerr=Deltayerr, fmt=".", color='black')
    

    if line_trend == 0:
        print('no trend line')

    if line_trend == 1:
        if len(GFAP_concentrations) > 1:
            # Select the concentration and Rct data for trendline fitting
            filtered_concentrations = GFAP_concentrations[trendline_start:trendline_end]
            filtered_Rct_values = Delta_Rct[trendline_start:trendline_end]
            
        if len(filtered_concentrations) > 1:  # Ensure there's enough data to fit
            coefficients = np.polyfit(filtered_concentrations, filtered_Rct_values, 1)  # Linear fit
            polynomial = np.poly1d(coefficients)
            trendline = polynomial(np.sort(filtered_concentrations))
            ax.plot(np.sort(filtered_concentrations), trendline, linestyle='--', color=colors[colornumber], label=f'{file_name} Trend line')

        # Calculate R^2
        y_pred = polynomial(filtered_concentrations)
        residuals = np.array(filtered_Rct_values) - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(filtered_Rct_values) - np.mean(filtered_Rct_values))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Display trendline equation and R^2
        trendline_eq = f'Rct = {coefficients[0]:.2f} * [GFAP] + {coefficients[1]:.2f}'
        print(f'Trendline equation {file_name}: {trendline_eq}')
        print(f'R^2: {r2:.4f}')

def calculate_error(error_r1, error_r2, r1, r2):
    import math
    
    # Calculate the error associated with Δr
    error_delta_r = (1 / r2) * math.sqrt(error_r1**2 + ((r1 / r2) * error_r2)**2)
    return error_delta_r


# The folowing def its to plot the Open Circuit Potencimetry of pseudo reference electrodes with the example 
# costumized to LIG vs SPE carbon

def OPC(Directory,finalDirectory,file1,type,file2,Probe):
    '''  name = Ref_LIG
    probe = PBS0_01_mM
    type = 8_11
    '''
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os


    os.chdir(Directory)

    LIG = f'{file1}_{Probe}_{type}.xlsx'

    Commercial = f'{file2}_{Probe}.xlsx'

    lig = pd.read_excel(LIG)
    lig_header = lig.iloc[0]
    lig=lig[1:]
    lig.columns = lig_header
    lig.head()

    commercial = pd.read_excel(Commercial)
    commercial_header = commercial.iloc[0]
    commercial=commercial[1:]
    commercial.columns = commercial_header
    commercial.head()


    fig, ax = plt.subplots(figsize =(10, 7))
    ax = plt.plot(lig["s"], lig["V"], label='LIG Pseudo-RE', color='#00ff00')

    ax = plt.plot(commercial["s"], commercial["V"], label='SPE Pseudo-RE', color='#e4007c')

    plt.xlabel("Time / s",fontsize=12, fontweight='bold')
    plt.ylabel("OPC / V",fontsize=12, fontweight='bold')
    plt.title(f"Stabilaty of Pseudo-Reference Electrode",fontweight='bold')
    plt.legend(prop={'size': 8.5,'weight': 'bold'})    
    
    os.chdir(finalDirectory)
    
    plt.savefig(f"Stabilaty of Pseudo-Reference Electrode.png")


# now to the CV files with the objective to evaluate the best potencial to use on electrodeposition 

def CV_evaluateCA_potencia(Directory,savefiledirectory,type):
    '''
    type: it's the type of electrode you are using for example CSPE for Carbon Screen Printed Electrods 
    '''
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os


    os.chdir(Directory)

    _0_2file = f'0_2_{type}.xlsx'

    _0_6file = f'0_6_{type}.xlsx'

    _1_0file = f'1_0_{type}.xlsx'

    _1_2file = f'1_2_{type}.xlsx'

    _1_4file = f'1_4_{type}.xlsx'

    _1_5file = f'1_5_{type}.xlsx'

    _1_6file = f'1_6_{type}.xlsx'

    _1_8file = f'1_8_{type}.xlsx'

    _0_2 = pd.read_excel(_0_2file)
    _0_2_header = _0_2.iloc[0]
    _0_2=_0_2[1:]
    _0_2.columns = _0_2_header
    _0_2.head()

    _0_6 = pd.read_excel(_0_6file)
    _0_6_header = _0_6.iloc[0]
    _0_6=_0_6[1:]
    _0_6.columns = _0_6_header
    _0_6.head()

    _1_0 = pd.read_excel(_1_0file)
    _1_0_header = _1_0.iloc[0]
    _1_0=_1_0[1:]
    _1_0.columns = _1_0_header
    _1_0.head()

    _1_2 = pd.read_excel(_1_2file)
    _1_2_header = _1_2.iloc[0]
    _1_2=_1_2[1:]
    _1_2.columns = _1_2_header
    _1_2.head()
    
    _1_4 = pd.read_excel(_1_4file)
    _1_4_header = _1_4.iloc[0]
    _1_4=_1_4[1:]
    _1_4.columns = _1_4_header
    _1_4.head()

    _1_5 = pd.read_excel(_1_5file)
    _1_5_header = _1_5.iloc[0]
    _1_5=_1_5[1:]
    _1_5.columns = _1_5_header
    _1_5.head()
    
    _1_6 = pd.read_excel(_1_6file)
    _1_6_header = _1_6.iloc[0]
    _1_6=_1_6[1:]
    _1_6.columns = _1_6_header
    _1_6.head()

    _1_8 = pd.read_excel(_1_8file)
    _1_8_header = _1_8.iloc[0]
    _1_8=_1_8[1:]
    _1_8.columns = _1_8_header
    _1_8.head()

    fig, ax = plt.subplots(figsize =(10, 7))
    ax = plt.plot(_0_2["V"],_0_2["µA"], label=' -0.2 V CA', color='#e4007c')

    ax = plt.plot(_0_6["V"],_0_6["µA"], label=' -0.6 V CA', color='#556b2f')

    ax = plt.plot(_1_0["V"],_1_0["µA"], label=' -1.0 V CA', color='#0073cf')

    ax = plt.plot(_1_2["V"],_1_2["µA"], label=' -1.2 V CA', color='#CC6CE7')

    ax = plt.plot(_1_4["V"],_1_4["µA"], label=' -1.4 V CA', color='orange')

    ax = plt.plot(_1_5["V"],_1_5["µA"], label=' -1.5 V CA', color='yellow')

    ax = plt.plot(_1_6["V"],_1_6["µA"], label=' -1.6 V CA', color='#9B08D1')

    ax = plt.plot(_1_8["V"],_1_8["µA"], label=' -1.8 V CA', color='#9999FF')

    plt.xlabel("Current / µA",fontsize=12, fontweight='bold')
    plt.ylabel("Potential / V",fontsize=12, fontweight='bold')
    plt.legend(prop={'size': 8.5,'weight': 'bold'})    
    
    os.chdir(savefiledirectory)
    
    plt.savefig(f"Current_vs_Potenctial_{type}.png")

def Analyse_Data_Desktop_Teste_Lenovo(Directory,savefiledirectory):
    '''
    This is a fuction dedicated to LENOCO ideaped with the focus to Extrat Xlsx of EIS meaasurments,
      fitings, Rs and Rct form .pssession files 
    '''
    import os 
    import fnmatch
    import PStracelib as PS
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt 
 
    list = []
    for file_name in os.listdir(Directory):
        if fnmatch.fnmatch(file_name,'*.pssession'):
                list.append(file_name[:-10])


    done = []
    for file_name in os.listdir(savefiledirectory):
        if fnmatch.fnmatch(file_name,'*.txt'):
                done.append(file_name[:-4])

    already_done=[]
    to_do= []
    for file in list: 
        if file in done:
            already_done.append(file)
        else:   
            to_do.append(file)

    order = 0
    for name in to_do:
        order = order + 1
    print(f'The number of files to work is {order} the total time will be ~{order * 5} min')


    n=0
    for name in to_do:
        n = n +  1
        percentage = (n * 100)/ order
        print(f'{n}º ------> {name}')
        print(f'Loading... {percentage} %')
        print(f'It will take more {(order*5-n*5)/60} h ')
        PS.Lenovo(name)
        print('___________________________________________________________________________________________________________________')


# Changing the name of files to the standart in this file
def Standart_Not_Clean_pssession (Directory,Probe):
    import os 
    import fnmatch
    import PStracelib as PS
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    list = []
    for file_name in os.listdir(Directory):
        if fnmatch.fnmatch(file_name,f'*.pssession'):
                list.append(file_name)

    for a in list:
        Type = a.split('_')[1]
        Name = a.split('_') [0].split('.')[0]
        if 'not' in Type:
            old_file = os.path.join(Directory, a)
            new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.pssession')
            os.rename(old_file, new_file)
        if 'NOT' in Type:
            old_file = os.path.join(Directory, a)
            new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.pssession')
            os.rename(old_file, new_file)
        if 'Not' in Type:
            old_file = os.path.join(Directory, a)
            new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.pssession')
            os.rename(old_file, new_file)

def Standart_Not_Clean (Directory,Probe,Type):
    import os 
    import fnmatch
    import PStracelib as PS
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    list = []
    for file_name in os.listdir(Directory):
        if fnmatch.fnmatch(file_name,f'*.{Type}'):
                list.append(file_name)
    print(list)
    for a in list:
        Type = a.split('_')[1]
        Name = a.split('_') [0].split('.')[0]
        fit = a.split('_') [-1].split('.')[-2]
        
        if 'not' in Type:
            if 'fit' not in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)
                
        if 'NOT' in Type:
            if 'fit' not in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)
                
        if 'Not' in Type:
            if 'fit' not in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)

def Standart_Not_Clean_fit (Directory,Probe,Type):
    import os 
    import fnmatch
    import PStracelib as PS
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    list = []
    for file_name in os.listdir(Directory):
        if fnmatch.fnmatch(file_name,f'*.{Type}'):
                list.append(file_name)
    print(list)
    for a in list:
        Type = a.split('_')[1]
        Name = a.split('_') [0].split('.')[0]
        fit = a.split('_') [-1].split('.')[-2]
        
        if 'not' in Type:
            if 'fit' in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)
                
        if 'NOT' in Type:
            if 'fit' in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)
                
        if 'Not' in Type:
            if 'fit' in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_Not_Clean_{Probe}.{Type}')
                os.rename(old_file, new_file)
                
def Standart_gold(Directory,Probe,Type):
    import os 
    import fnmatch
    import PStracelib as PS
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    list = []
    for file_name in os.listdir(Directory,Probe,Type):
        if fnmatch.fnmatch(file_name,f'*.{Type}'):
                list.append(file_name)
    print(list)
    for a in list:
        Type = a.split('_')[1]
        Name = a.split('_') [0].split('.')[0]
        fit = a.split('_') [-1].split('.')[-2]
        print(fit)
        if 'gold' in Type:
            print(a)
            if 'fit' not in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_gold_{Probe}.{Type}')
                os.rename(old_file, new_file)

    list = []
    for file_name in os.listdir(Directory,Probe,Type):
        if fnmatch.fnmatch(file_name,f'*.{Type}'):
                list.append(file_name)
    print(list)
    for a in list:
        Type = a.split('_')[1]
        Name = a.split('_') [0].split('.')[0]
        fit = a.split('_') [-1].split('.')[-2]
        print(fit)
        if 'gold' in Type:
            print(a)
            if 'fit' in fit:
                old_file = os.path.join(Directory, a)
                new_file = os.path.join(Directory, f'{Name}_gold_{Probe}.{Type}')
                os.rename(old_file, new_file)
