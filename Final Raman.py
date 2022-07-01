import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import glob
import os
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from io import StringIO
import scipy as scipy
from natsort import natsorted
#Appearance stuff (colors, font size etc)

#My colors
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5'])
#                                                       0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry3


# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#e41a1c','#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#dede00'])

# Nature colors
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['0C5DA5', 'FF2C00', 'FF9500', '00B945', '845B97', '474747', '9e9e9e'])

#Seaborn colors
# colors=sns.color_palette("rocket",3)

# Font size. Dictionary taking numerous parameters.
plt.rcParams.update({'font.size' : 12})


# Gets current working directory (cwd)
cwd=os.getcwd()

#Creates a folder to store the graphs inside the cwd
# mesa= cwd+ '\\Neat PLA\\'
# if not os.path.exists(mesa):
#     os.makedirs(mesa)

#Path to data. Full path if in different folder than this script is. Otherwise *files common name part*".
file_list = [i for i in glob.glob(r"0.25mm*")]
#Sort in order of number appearing in front. Omit or adjust accordingly in case of different naming.
file_list=natsorted(file_list)

# Loading all the csv files to create a list of data frames
data = [pd.read_csv(file,names=["Wavenumber","Intensity"], skiprows=1,delimiter='	') for file in file_list]

#Replaces the useless part of the dataframes' names to auto-generate a better suited legend.
file_list=[file.replace("0.25mm\\", '') for file in file_list]
file_list=[file.replace('.txt','') for file in file_list]
file_list=[file.replace('_',' #') for file in file_list]


#Markers tuple to get different marker for each line
markers=('o', 's', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

from scipy.linalg import cholesky
def arpls(y, lam=1e4, ratio=0.05, itermax=100):

    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

#Plot in the same graph. Can as well locate columns with name but .iloc is useful af. Do as I say, not as I do.
for index, dataframe in enumerate(data):
    fig = plt.figure(figsize=(5,5),dpi=300)
    baseline = arpls(data[index]['Intensity'][:], 1E6, 0.001)
    baseline_subtracted = data[0]['Intensity'][:] - baseline
    plt.plot(data[index]["Wavenumber"],baseline_subtracted,label=file_list[index],linestyle='-', linewidth=1,) #marker=markers[index], mfc='w',
    
    # Axes limits. Adjust accordingly.
    # plt.xlim([0.0012,0.009])
    # plt.ylim([-2,15])
    
    # Ticks
    plt.minorticks_on() #uses minor ticks
    # plt.tick_params(direction='in',right=False, top=False) #Obsolete with following lines. Places MAJOR TICKS only on sides & handles direction. Avoid using it.
    # plt.tick_params(labelsize=10) #Changes values' font, which we locked earlier.
    # plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True) #Turns on/off axis values on all 4 sides
    # xticks = np.arange(1, 100,10) #Numpy to place ticks in certain positions X axis
    # yticks = np.arange(0,160.1,4) #Numpy to place ticks in certain positions Y axis
     
    plt.tick_params(direction='in',which='minor', length=2, bottom=True, top=False, left=True, right=False) #Properly handles minor ticks
    plt.tick_params(direction='in',which='major', length=4, bottom=True, top=False, left=True, right=False)# Bot/left by default True if memory serves me. All change with =False though.
    #plt.xticks(xticks) #Plots ticks
    #plt.yticks(yticks) #Plots ticks
    
    #Plots grid in the background
    plt.grid(True,linestyle='dashed', linewidth='0.3', color='grey',alpha=0.8)
    
    # Axis labels
    plt.xlabel(r'Wavenumber (cm$^{-1}$)')
    plt.ylabel(r'Intensity (a.u.)')
    
    # Legend
    plt.legend(loc=('upper right'),frameon=False)  # Adds the legend. # loc='center',bbox_to_anchor=(1.2,0.5),


#Peak fitting
baseline = arpls(data[index]['Intensity'][:], 1E6, 0.001)
baseline_subtracted = data[0]['Intensity'][:] - baseline
y_array=baseline_subtracted
x_array=data[0]['Wavenumber'].copy()

def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)

#2 lorentzian in reality
def _3Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2))

popt_3lorentz, pcov_3lorentz = scipy.optimize.curve_fit(_3Lorentzian, x_array, y_array, p0=[400, 1300, 100,500, 1500, 100])

perr_3lorentz = np.sqrt(np.diag(pcov_3lorentz))

pars_1 = popt_3lorentz[0:3]
pars_2 = popt_3lorentz[3:6]
lorentz_peak_1 = _1Lorentzian(x_array, *pars_1)
lorentz_peak_2 = _1Lorentzian(x_array, *pars_2)

plt.plot(x_array, _3Lorentzian(x_array, *popt_3lorentz), 'r--', linewidth=0.7)

# peak 1
plt.plot(x_array, lorentz_peak_1, "g",linestyle='-', linewidth=1)
plt.fill_between(x_array, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)

# peak 2
plt.plot(x_array, lorentz_peak_2, "y",linestyle='-', linewidth=1)
plt.fill_between(x_array, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)
