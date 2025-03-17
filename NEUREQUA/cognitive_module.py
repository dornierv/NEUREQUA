# Import librairies
import random
import neo.rawio
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb
import os
import neo
import mne
import pandas as pd
import scipy.stats as stats
import matplotlib





def create_epoch(Folder,t_min,t_max):
    """
    Create an Epoch structure in MNE based on the Events registered in the .nev of Neuralynx acquisition system
    To load the data from the .nev file I used the following library:
    https://github.com/alafuzof/NeuralynxIO


    Parameters
    ---------------------------
    Folder : string
        Path where your ncs files and the Events.nev files are stored
    
    t_min : float
        Time to include in the baseline (before the onset of event)
    
    t_max : float
        Time to include after the onset of the event
    """
    import neuralynx_io

    raw_data = mne.io.read_raw_neuralynx(Folder,exclude_fname_patterns=list(['*'+'_sub.ncs']))

    # Load the events.nev
    nev = neuralynx_io.load_nev(Folder+'./Events.nev')  # Load event data into a dictionary


    # Only keep events that are not fixation cross
    ttl_vals = nev['events']['ttl']

    idx_trials = np.nonzero(nev['events']['ttl'])



    # Get the timestamps of each trials
    ts_trials = nev['events']['TimeStamp'][idx_trials]


    # Get the timestamps relative to the onset of the recording
    ts_relatif = ts_trials - nev['events']['TimeStamp'][0]


    # Timestamps are expressed in micro-seconds in Neuralynx so divide by 10^6
    ts_second = ts_relatif / 1000000


    # Transform into samples
    onset_sample = ts_second*32768

    # Create structure (nEvents,3) to use it with MNE to create an Epoch object
    event = np.array((onset_sample,np.zeros(len(onset_sample)), ttl_vals[idx_trials])).astype(int)


    # Transpose from (3,nEvents) to (nEvents,3)
    event = event.T

    # Create event object
    event_object = mne.Epochs(raw_data, events=event,tmin=t_min,tmax=t_max,baseline=None,preload=False)

    #mean_epoch = event_object.average()

    return event_object







def plot_artefact_map(epoch_data,path):
    """
    Plot figure to show the variance of each trial and each channel
    Enables us to quickly see the channels that are artefacted (e.g., by epileptic activities)
    and also trials contaminated

    Just like the figure9.B of Mercier et al. (2022)

    Parameters
    ---------------------------
    epoch_data : array
        Matrice with the following shape (nTrials, nChannels, nSamples)

    path : String
        Path where you want to store results of this analyses

    Returns
    ---------------------------
    Matplotlib plot containing heatmap and variance of each channels for each trials
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)
    
    var_ch = list()

    nCh = epoch_data.shape[1]

    for iCh in range(epoch_data.shape[1]):
        data_channel = epoch_data[:,iCh,:]

        # Compute the variance for each trial
        variance_trial = np.var(data_channel,1)
    
        var_ch.append(variance_trial)
   

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(4,4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, :3])
    y_hist = fig.add_subplot(grid[:-1:, 3:], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, :3], yticklabels=[], sharex=main_ax)

    # scatter points on the main axes
    sb.heatmap(var_ch,ax=main_ax,cbar=False,cmap="rocket_r") # pour l'instant rocket_r est la mieux
    main_ax.axes.get_xaxis().set_visible(False)
    main_ax.locator_params(axis='y',nbins=int(nCh/4+1)) 

    # histogram on the attached axes
    x_hist.plot(np.mean(var_ch,0),'.',color='coral')
    # Setting the number of ticks 
    x_hist.locator_params(axis='x',nbins=10) 
    x_hist.set_title('# Trials',loc='center')

    y = np.arange(len(var_ch))
    y_hist.plot(np.mean(var_ch,1),y,'.',color='coral')
    y_hist.axes.get_yaxis().set_visible(False)
    y_hist.set_title('# Channels',loc='center')


    plt.savefig(path+'Artefact_Map.png',transparent=True)

    plt.show()

    plt.close()
