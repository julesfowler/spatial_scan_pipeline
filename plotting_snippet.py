
## -- IMPORTS
import numpy as np
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.pyplot as plt

## -- PLOT
rc('figure', figsize=[15, 45])

fig = plt.figure()

axs = gridspec.GridSpec(12, 3, width_ratios=[2, 2, 1])

fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)

for index,key in enumerate(keys):
    bin_dict = data[key]
    coeffs_p = bin_dict['coeffs']
    coeffs_p['t0'] = t0
    coeffs_p['shift'] = shift
    t = np.array(bin_dict['t'])
    y = np.array(bin_dict['y']) 
    mod_t = np.linspace(np.min(t), np.max(t), 200)
    transit_y = fit_light_curve.batman_transit(coeffs_p, t, planet_parameters)
    transit_mod = fit_light_curve.batman_transit(coeffs_p, mod_t, planet_parameters)
    ramp = fit_light_curve.ramp(coeffs_p, t, planet_parameters)
    t = t - t0
    mod_t = mod_t - t0
    corr = y/ramp
    y /=y[21]
    resid = (y-transit_y)/np.max(y)*1e6
    resid_corr = (corr - transit_y)/np.max(y/ramp)*1e6
    
    ax1 = plt.subplot(axs[index*3])
    ax2 = plt.subplot(axs[index*3+1])
    ax3 = plt.subplot(axs[index*3+2])
    ax1.plot(mod_t, transit_mod, '--', color='grey')
    ax1.scatter(t, y, color=DGREEN, s=10, alpha=.8)
    ax1.set_xlim(-.11, .13)
    ax1.set_ylim(.989, 1.0035)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2.plot(mod_t, transit_mod, color='grey')
    ax2.scatter(t, corr, color=DBLUE, alpha=.8, s=10)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlim(-.11, .13)
    ax2.set_ylim(.989, 1.0035)
    
    ax3.hist(resid, bins=20, color=DGREEN, hatch='/', alpha=.4, edgecolor='black', orientation='horizontal')
    ax3.hist(resid_corr, bins=9, color=DBLUE, hatch='.', alpha=.4, edgecolor='black', orientation='horizontal')
    ax3.set_yticks(np.array([-2000, 0, -2000]))
    ax3.yaxis.tick_right()
    ax3.set_xlim(.1, 15) 
    plt.setp(ax3.get_xticklabels(), visible=False)
    
plt.setp(ax3.get_xticklabels(), visible=True)
plt.setp(ax2.get_xticklabels(), visible=True)
plt.setp(ax1.get_xticklabels(), visible=True)
plt.savefig('test.png')

