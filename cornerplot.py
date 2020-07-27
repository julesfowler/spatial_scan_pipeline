import corner, sys
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rcParams
import mpld3
from pdb import set_trace
#from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter

# Takes the .npz files of MC3 and an array of strings with the names
# of the jump parameters.

plt.ioff()
#plt.rcParams['xtick.major.size'] = 50
#plt.rcParams['axes.titlepad'] = 5
plt.rcParams['axes.formatter.useoffset'] = False

def cornerplot(chains, titles, truths, fileout):

    print('Plotting corner plot...')

    title_keys             = {}
    title_keys['fontsize'] = 18
    title_keys['loc']      = 'left'
    title_keys['va']       = 'bottom'
    label_keys             = {}
    label_keys['fontsize'] = 18
    label_keys['labelpad'] = 16

    hist_keys = {}
    hist_keys['log'] = False

    #rr = [(0.16, 0.17), 0.95, 0.99, 0.99, (-40, 0.), (0., 100.), (0., 0.002), .9, .99, .99] #*chains.shape[1]
    #rr = np.concatenate((np.asarray([0.98]), [0.99999]*(chains.shape[1] - 2), np.asarray([0.98])))
    #rr = [(0.163, 0.172), (0.062, 0.0797), 0.999, 0.999, 0.999, 0.999, 0.999, (-4, -3)]
    rr = [0.99]*chains.shape[1]
    #rr = [(0.155, 0.175), (0.0695, 0.075), (-10, -5.5), (-17, -10)]
    #fig, axes = plt.subplots(chains.shape[1], chains.shape[1], figsize=(20, 20))
    corner.corner(chains, labels = titles, use_math_text = True, quantiles=[.16,0.5,.84], show_titles = 'False', title_fmt = ".2f", title_kwargs = title_keys, label_kwargs = label_keys, hist_kwargs = hist_keys, smooth = False, range = rr, plot_datapoints = True, max_n_ticks = 3, truths = truths)
    '''
    fig = plt.gcf()
    axs = fig.get_axes()
    for ax in axs:
        ax.tick_params(labelsize = 16)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        for tick in ax.get_yticklabels():
            tick.set_rotation(70)
    fig.subplots_adjust(wspace = None, hspace = None, left=0.05, bottom=None, right=None, top=0.95)
    '''
    plt.savefig(fileout)

    return
