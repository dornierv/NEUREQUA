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


def load_raw_data(path,dtype,length,*args):
    """
    Load data from the raw files (e.g., .ncs for Neuralynx)

    Returns two objects :
        1) A raw-object from MNE (see https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw for documentation)

        2) An array containing raw values of the signal you want to analyze 

    Parameters
    ---------------------------
    path : string
        A string containing the path where your data are stored

    dtpe : string
        For Neuralynx data : 'ncs'
        For Blackrock data : 'nsX'
        For Dark Horse Neuro data : 'med'

    length: int or 'all'
        If you specify length with a int (e.g., 5) it will select randomly 5 minute of signal
        If you specify 'all' it will take the entire recording (can be slower, depends on your computational power)
    
    
    Returns
    ---------------------------
    raw_data : raw data in FIF format (see MNE)
        This object contains all informations about your data (channels name, sampling rate etc.)
    
    data_sig : np array
        Array containing raw values of the signal you want to load
        Will have a shape of nCh x nSamples
   """
    
    # Load 'ncs' from Neuralynx
    if dtype=='ncs':
        # Exclude files containing 'sub' because correspond to macro-electrodes
        raw_data = mne.io.read_raw_neuralynx(path,exclude_fname_patterns=list(['*'+'_sub.ncs']))
    # Load 'ns5' from Blackrock    
    elif dtype=='nsX':
        raw_data = mne.io.read_raw_nsx(path)
        # Exclude the reference channel
        raw_data.drop_channels('ainp1')
    elif dtype=='dat':
        # Here not sure everyone is using int16 but for us it is ok
        data_type = np.int16

        # In order to read the dat we absolutely need the number of channels so if not specified by user return a message
        if len(args)==0:
            print("Please indicate the number of channels in your dat file to load it")

        # Here we load data
        elif len(args)==1:
            size = os.path.getsize(path)
            size = int(size/np.dtype(data_type).itemsize)
            raw_data = np.memmap(path, mode='r', dtype=data_type, order='F', shape=(args[0], int(size/args[0])))

        # If the user specify too much arguments we ask him to only add the number the channels and nothing more    
        else:
            print("You specify too much arguments please only indicate the number of channels")
    
    elif dtype == 'med':
        # Here we create a raw object from the med folder using this: 
        # https://mne.tools/stable/auto_examples/io/read_neo_format.html

        

        # Read data from the .med folder
        reader =  neo.io.MedIO(path)


        # Get the first block - proxy (do not load in memory)
        block = reader.read(lazy=True)[0]

        # Get a proxy data from first segment (you have to get only one segment)
        segment = block.segments[0]

        # Get the signal from all channels
        signals_proxy = segment.analogsignals[0]
    
    else:
        print("File format not supported, you can load .ncs, .nsX, .med for now")
        print("Contact us to add new file format")


        
    
    # If you want to load all recording
    if length=='all':
        # If want to load all file then load all med recording
        if dtype == 'med':
            
            signals = signals_proxy.load()

            data = signals.rescale("V").magnitude.T
            sfreq = signals.sampling_rate.magnitude


            # Get the name of the channels
            ch_names = [f"{reader.header['signal_channels'][idx][0]}" for idx in range(signals.shape[1])]
            # Attribute a type to channels (here eeg)
            ch_types = ["eeg"] * len(ch_names)  # if not specified, type 'misc' is assumed

            # Create a raw object in MNE
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
            raw_data = mne.io.RawArray(data, info)

            raw_interest = raw_data.crop(tmin=0)

            data_sig = raw_interest.get_data()
        else:
            raw_interest = raw_data.crop(tmin=0)

            data_sig = raw_interest.get_data()
    # Else will randomly choosed a portion of length specified in input
    else:

        if dtype == 'med':
            import random
            # Determine the index max for the length you want
            # e.g. If you want 5 minutes of signal the last sample can not be less than 5 minutes before the end of your recording
            sampling_rate = int(signals_proxy.sampling_rate)
            last_sample = int(signals_proxy.duration*sampling_rate)
            

            idx_max_random = int(last_sample - (sampling_rate*60*length)) # (sr*60*lenght) correspond to the length of your subselection


            # Here randomly select the starting index of the 5 minutes
            idx_time = int(random.random()*idx_max_random)

            

            signal_crop = signals_proxy.time_slice(t_start=idx_time/sampling_rate,t_stop=(idx_time/sampling_rate)+length*60)

            data = signal_crop.rescale("V").magnitude.T
            sfreq = signal_crop.sampling_rate.magnitude


            # Get the name of the channels
            ch_names = [f"{reader.header['signal_channels'][idx][0]}" for idx in range(signal_crop.shape[1])]
            # Attribute a type to channels (here eeg)
            ch_types = ["eeg"] * len(ch_names)  # if not specified, type 'misc' is assumed

            # Create a raw object in MNE
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
            raw_data = mne.io.RawArray(data, info)

            # Load data in memory
            data_sig = raw_data.get_data()
        
        
        else:
            # Select randomly the first sample 
            idx_time = random_time(raw_data,raw_data.info['sfreq'],length)
            sampling_rate = raw_data.info['sfreq']
            # Crop the signal between 1st sample and last sample (last sample - 1st sample = length)
            raw_interest = raw_data.crop(tmin=idx_time/sampling_rate,tmax=(idx_time/sampling_rate)+length*60)

            # Load data in memory
            data_sig = raw_interest.get_data()

    
    return raw_data,data_sig




def splitCharNum(string) :
    """
    Separate a string containing characters and numbers in two objects

    Parameters
    ---------------------------
    string : string
        String containing characters and numbers (e.g., 'da1')

    Returns
    ---------------------------
    char : string
        string containing only the characters from the input (e.g., 'da')

    number : string
        string containing only the numbers from the input (e.g., '1')
    """
    import re
    char,number=re.findall(r'[A-Za-z]+|\d+', string)
    return char,number




def get_unique_unsorted(array):
    """
    Maintains the order of appearance in the original array
    Does not sort by ascending order (figures) or alphabetic order (characters)


    Parameters
    ---------------------------
    array : array of string
        Array of string containing the name of your channels (e.g. ['da','db'])
    
        
    Returns 
    ---------------------------
    uniqueUnsorted : string
        array of string containing 
    """
    unique, uniqueInds=np.unique(array, return_index=True)
    uniqueUnsorted=array[np.sort(uniqueInds)]
    return uniqueUnsorted




def reorder_data(Regions,data):
    '''
    When there is 3 tetrodes on the same shaft order will be x1, x10, x11, x12, x2, x3 ... x9 on the raw file

    We want to re-organize it so the order is x1, x2, x3 ... x9, x10, x11, x12

    Will return the array of regions name and data in the right order


    Parameters
    ---------------------------
    Regions : array of string
        Array of strings containing all the name of the regions implanted with your channels
    
    data : array
        Matrix containing raw values with shape nChannels x nSamples
    
        
    Returns
    ---------------------------
    Regions_ok : array of string
        Array of strings containing the name in the right order (e.g., 'x1', 'x2', 'x3' ... 'x9', 'x10', 'x11', 'x12')
    
    data_ok : array
        Matrix with shape nChannels x nSamples but re-organize to match the order of channel labels
    '''
    chChar=np.array([splitCharNum(reg)[0] for reg in Regions])
    chNumber=np.array([splitCharNum(reg)[1] for reg in Regions])
    regInds=np.array([chChar==char for char in get_unique_unsorted(chChar)])
    

    argsortChRegs=np.concatenate([np.where(regInds[regi])[0][np.argsort(np.array(chNumber[regInds[regi]], int))]\
         for regi in range(regInds.shape[0])])
    

    Regions_ok = Regions[argsortChRegs]
    data_ok = data[argsortChRegs]

    return Regions_ok,data_ok    






def find_nearest(array, value):
    """
    Find the index where there is the nearest values in an array from the one we want


    Parameters
    ---------------------------
    array : np.array
        array containing data (e.g., [2, 7, 12])

    value : int
        The value we want to find the closest (e.g. 11)

    Returns
    ---------------------------
    idx : int
        Will return the index  in array closest to value (e.g., 2)
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def random_time(data,sampling_rate,length):
    """
    Select randomly length minutes of signal from your entire recoding

    
    Parameters
    ---------------------------
    data : The raw object of data (Raw object from MNE)
    
    sampling rate : int
        The sampling rate of your recording (e.g., 32768)
    
    length : int
        Length in minute of the signal you want to analyze
    
        
    Returns 
    ---------------------------
    idx_debut : int
        Correspond to the first sample of the signal, the beginning of the recording you want
        to analyze
    """
    
    # Determine the index max for the length you want
    # e.g. If you want 5 minutes of signal the last sample can not be less than 5 minutes before the end of your recording
    last_sample = data.last_samp.T

    idx_max_random = int(last_sample - (sampling_rate*60*length)) # (sr*60*lenght) correspond to the length of your subselection


    # Here randomly select the starting index of the 5 minutes
    idx_debut = int(random.random()*idx_max_random)

    return idx_debut



def p_welch(data,sr,sr_down,fr_low,fr_high):
    """
    Compute the power spectrum of your signal in order to identify frequencies present in your signal
    Here we use the welch's method to compute the power spectrum

    Parameters
    ---------------------------
    data : array
        Matrix containing data of one channel containing nSamples
    
    sr : int
        Sampling rate of your recording (e.g., 32768)
    
    sr_down : int
        Sampling rate downsample to get signal in lower frequencies (e.g., 8192 Hz)
    
    fr_low : int
        The lowest frequency from which we compute the power spectrum
    
    fr_high : int 
        The highest frequency from which we compute the power spectrum
    
    
    Returns
    ---------------------------
    pxx_log : array
        Array containing power spectrum values for the channel of interest
    
    f_plot : array
        Array containing the values of frequency associate with each power spectrum value (will be useful for the plot)
    """

    # Dowsample from sr to fs in order to speed up computing
    ds_factor = int(sr/sr_down) # Calcule downsample factor by dividing sampling rate (of the original signal) by the sampling rate desired in output

    #data_ds = sig.resample(data,int(data.shape[0]/ds_factor))
    
    # Compute welch method to estimate PSD
    f, pxx = sig.welch(data, fs=sr, nperseg=4096)


    # Get the index from fr_low to fr_high Hz to plot the results
    idx_fr_lim = find_nearest(f,fr_high)
    idx_debut = find_nearest(f,fr_low)

    pxx_log = 10*np.log10(pxx[idx_debut:idx_fr_lim])
    f_plot = f[idx_debut:idx_fr_lim]
    #pxx_log = pxx[idx_debut:idx_fr_lim]


    

    return pxx_log,f_plot



def plot_all_chan(f,nCh,chRegs,psd,bsnm,session,probe_type,saveFolder):
    """
    This function will plot the power spectrum of all micro-channels on the same plot

    Parameters 
    ---------------------------
    f : array
        Array containing frequency values (it is the output of p_welch)
    
    nCh : int
        Number of channels in your recording
    
    chRegs : array of strings
        Array containing name of your channels
    
    psd : array
        Array of power spectrum values for each value of f, it is the output of p_welch
    
    session : string
        Name of the session you analyze, specify at the beginning of the jupyter notebook

    probe_type : string
        String containing the model of micro-electrodes you have in your dataset (for now: Dixi or Ad-tech only)

    
    saveFolder : string
        String containing the path of the folder where you want to save figure

    Returns
    ---------------------------
    Matplotlib plot containing power spectrum of each channels and saved in the folder specified
    """


    # Check whether the specified path exists or not
    isExist = os.path.exists(saveFolder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(saveFolder)
   

    # List of colors, each tetrode will be in the same color but with different transparency
    colors = ['black','red','orangered','saddlebrown','gold','olive','chartreuse','turquoise','darkslategray','dodgerblue','midnightblue','slateblue','darkviolet','violet','magenta','crimson']
    
    if probe_type == 'Dixi':
        transparency = [0.4,0.6,0.8,1]
    elif probe_type == 'Ad-tech':
        transparency = [0.4,0.5,0.6,0.7,0.8,0.9,1]

    
    # Initialize to zero the current tetrode (so it takes the first one)
    iGroup = 0

    min_pwr = []
    max_pwr = []

    # Here we loop on all channel and plot the PSD corresponding to each channel
    for i in range(0,nCh):
        # Determine wich micro-wire of the tetrode it is (1st, 2nd, 3rd, 4th)
        if probe_type == 'Dixi':
            modulo_ch = i%4
        elif probe_type == 'Ad-tech':
            modulo_ch = i%8

        # Plot the PSD of the micro-wire with according alpha transparency
        plt.plot(f,psd[i],color=colors[iGroup],alpha=transparency[modulo_ch])


        # Keep min and max to automatically adjust the limit of the plot
        min_pwr.append(min(psd[i]))
        max_pwr.append(max(psd[i]))

        # When we did the last micro-wire of the tetrode iTetrode increment to go to the next tetrode
        if probe_type == 'Dixi':
            if modulo_ch==3:
                iGroup=iGroup+1
        elif probe_type == 'Ad-tech':
            if modulo_ch == 8:
                iGroup = iGroup + 1


    # Legend and save plot
    plt.ylabel('10 * log10(Power)')
    #plt.ylabel('PSD [V**2/Hz]')
    plt.xlabel('Frequency (Hz)')
    plt.title('Power Spectrum (Welch ''s method) - '+bsnm+' - '+session)


    #plt.ylim((min(min_pwr)-1,max(max_pwr)+1))
    plt.legend(loc='center left',labels=chRegs,bbox_to_anchor=(1,0.5),fontsize=3)
    plt.savefig(saveFolder + 'PSD_All_Channels_0_600Hz_'+bsnm+'_'+session+'.png', dpi=300)


    plt.show()







def plot_noise(data,sr,num_channel,name_channels,path,save=1,fr_low=300,fr_high=3000): #
    '''
    This function will plot the raw signal filtered between 300 and 300 Hz for 1 second randomly choosed in the 5 minutes window
    So we can have an idea of the level of noise during our recording


    Parameters 
    ---------------------------
    data : array
        Data from one particular channel

    sr : int
        sampling rate of the signal
    
    num_channel : int
        Number of the channel in your recording, used to save figure
    
    name_channels : string
        String containing the name of the channel you want to plot
    
    path : string
        String containing the path where you want to save the figures
    
    save : int, default = 1
        If you don't want to save figure put save = 0
    
    fr_low : int, default = 300
        The lowest frequency for your band pass filter 
    
    fr_high : int, default = 3000
        The highest frequency for your band pass filter
    
    Returns
    ---------------------------
    Matplotlib plot saved in the specific folder if you want to
    '''
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)
   


    # Select 1s of data
    idx_max_random = int(len(data) - (sr)) # Select the index so the beginning is at least 1s before the end of the recording

    
    # Here randomly select the starting index of the second
    idx_debut = int(random.random()*idx_max_random)
    
    

    # Design our filter
    sos = sig.butter(3,[fr_low,fr_high],'bandpass',fs=sr,output='sos')

    # Filter the data
    data_filtered = sig.sosfilt(sos,data)


    # Plot the data
    plt.figure(figsize=(12,5))
    plt.plot(np.linspace(0,1,sr),data_filtered[idx_debut:idx_debut+sr])
    plt.ylabel('µV')
    plt.xlabel('Time (s)')
    plt.title('Noise level - channel : '+ name_channels + ' (n° : '+str(num_channel)+')')

    if save==1:
        plt.savefig(path+'Noise_level_channel'+str(num_channel)+'.jpg')
    
    plt.close()

    

def plot_raw(data,sr,num_channel,path,save=1): 
    '''
    In entry take the 5 minutes of signal that were randomly choosed before
    This function will plot the raw signal for 1 second randomly choosed in the 5 minutes window

    Parameters 
    ---------------------------
    data : array
        data from one particular channel

    sr : int
        sampling rate of the signal
    
    num_channel : int
        Number of the channel in your recording
    
    path : string
        Path where you want to store your figure in output
    
    save : Boolean, default = 1 else put save = 0

    Returns
    ---------------------------
    Matplotlib plot with raw signal 
    '''
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)
   

    # Select 1s of data
    idx_max_random = int(len(data) - (sr)) # Select the index so the beginning is at least 1s before the end of the recording

    
    # Here randomly select the starting index of the second
    idx_debut = int(random.random()*idx_max_random)
    
    

    


    # Plot the data
    plt.figure(figsize=(12,5))
    plt.plot(np.linspace(0,1,sr),data[idx_debut:idx_debut+sr])
    plt.ylabel('µV')
    plt.xlabel('Time (s)')
    plt.title('Raw signal - channel : ' + str(num_channel))

    if save==1:
        plt.savefig(path+'Raw_Signal_channel'+str(num_channel)+'.jpg')
    
    plt.close()



def tblprep(path,electrodes,sub,sess) :
    '''
    here we prep the excel file where all RMS for a patient is stored
    path: path to an existing excel file
    electrodes: electrode names 
    sub: patient number
    sess: session number 
    '''
    
    #open the preexisting excel file
    import os.path as path_os
    
    if path_os.exists(path):
        tbl = pd.read_excel(path, header=0)
    else:
        # if empty, creat columns with named sub, session, and with electrode names
        tbl = pd.DataFrame()
        tbl['sub'] = 'DefaultValue'
        tbl['run'] = 'DefaultValue'
        tbl['electrodes'] = 'DefaultValue'
        tbl['RMS'] = 'DefaultValue'
        tbl['RMS_filter'] = 'DefaultValue'
        tbl['variance'] = 'DefaultValue'
        tbl['variance_norm'] = 'DefaultValue'
        tbl['tetrode_cor'] = 'DefaultValue'
        tbl['deviation'] = 'DefaultValue'
        tbl['kurtosis'] ='DefaultValue'
        tbl['region'] = 'DefaultValue'
        tbl['SNR'] = 'DefaultValue'
        tbl['Artefact'] = 'DefaultValue'
        tbl['Hurst'] = 'DefaultValue'
    
    #find the first empty line
    if len(tbl) == 0:
        i=0
        
    else :
        i=0
        while i < len(tbl) and pd.notna(tbl.iloc[i, 0]) and tbl.iloc[i, 0] != "":
            i = i + 1
            
    #write sub, session and electrode names 
    for j in range(i,i+len(electrodes)): 
        x = j - i
        tbl = pd.concat([tbl, pd.DataFrame([{'sub': sub, 'run': sess,'electrodes': electrodes[x]}])], ignore_index=True)

        
    #save and replace the old file
    tbl.to_excel(path, index=False)

  

def rms_signal (data,path,electrodes,sub,sess) :
    '''
    here we take the 5 minutes of signal that were randomly choosed before
    This function will calculate the RMS (root mean square) for 1 second randomly choosed in the 5 minutes window

    data : data from one particular channel

    sr : sampling rate of the signal
    '''
    # Calculate the Root Mean Square (RMS) on the whole data
    rms = np.sqrt(np.mean(data**2)) 
    
    #open the table
    tbl = pd.read_excel(path)

    #write in the table
    tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == electrodes), 'RMS'] = rms

    #save the table
    tbl.to_excel(path, index=False)

    return tbl


def rms_signal_filtered (data,path,electrodes,sub,sess,fr_low,fr_high,sr) :
    '''
    here we take the 5 minutes of signal that were randomly choosed before
    This function will calculate the RMS (root mean square) for 1 second randomly choosed in the 5 minutes window

    data : data from one particular channel

    sr : sampling rate of the signal
    '''
    # Band-pass filter between two frequencies
    # Design our filter
    sos = sig.butter(3,[fr_low,fr_high],'bandpass',fs=sr,output='sos')

    # Filter the data
    data_filtered = sig.sosfilt(sos,data)


    # Calculate the Root Mean Square (RMS) on the whole data
    rms = np.sqrt(np.mean(data_filtered**2)) 
    
    #open the table
    tbl = pd.read_excel(path)

    #write in the table
    tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == electrodes), 'RMS_filter'] = rms

    #save the table
    tbl.to_excel(path, index=False)

    return tbl

    
def plot_rms(pathtbl,savingpath,sub,sess,save=1):
    tbl = pd.read_excel(pathtbl)
    rmstbl = tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess), ['sub', 'run', 'electrodes', 'RMS']]
        
    plt.rcParams.update({'font.size': 11})
    plt.figure(figsize=(20,5))
    plt.title('RMS for each channel')
    plt.xticks(rotation = 45)

    sb.lineplot(x='electrodes', y='RMS', data=rmstbl)    
    
    if save==1:
        plt.savefig(savingpath+'RMS_AllMicro.jpg')



def correlation_coefficient(data,chRegions,path,pathtbl,sub,sess,probe_type,save=1):
    '''
    This function compute a correlation coefficient between one micro-wire and all the other micro-wires
    from the same tetrode

    see Tuyisenge, V., Trebaul, L., Bhattacharjee, M., Chanteloup-Forêt, B., Saubat-Guigui, C., Mîndruţă, I., Rheims, S., 
    Maillard, L., Kahane, P., Taussig, D., & David, O. (2018). 
    Automatic bad channel detection in intracranial electroencephalographic recordings using ensemble machine learning. 
    Clinical Neurophysiology, 129(3), 548‑554. https://doi.org/10.1016/j.clinph.2017.12.013


    Parameters
    ---------------------------

    data: array
        Matrix with your data with shape nChannels x nSamples

    chRegions: Array
        Vector with the name of the micro-wires

    path: string
        Path where you want to save the figure obtained
    
    pathtbl : string
        Path where you store the excel file containing all values for the patient you analyze
    
    sub : string
        String like 'sub-XX' with XX being the number of the subject in your database
    
    sess : string
        String containing the name of the session you are analyzing (specified at the beginning of the jupyter notebook)

    probe_type : string
        String containing the model of micro-electrodes you have in your dataset (for now: Dixi or Ad-tech only)

    save=1 (default) to save the figure, put save = 0 if you don't want to save the figure

    Returns
    ---------------------------
    Matplotlib plot with correlation coefficient for each channel
    Values of the correlation coefficient saved in an excel fil to get a report of the analyzes
    '''
    import os
      
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)


    mean_corr = list()

    # Loop over each tetrode
    if probe_type == 'Dixi':
        for iTetrode in range(int(data.shape[0]/4)):

            # Select data for the tetrode of interest
            data_tetrode = data[iTetrode*4:iTetrode*4+4]


            # For each channel in the tetrode of interest
            for i in range(data_tetrode.shape[0]):
                corr = np.zeros(data_tetrode.shape[0])
                for j in range(data_tetrode.shape[0]):
                    if i!=j:
                        # Compute the pearson correlation between channel i and j
                        corr[j] = stats.pearsonr(data[i+iTetrode*4],data[j+iTetrode*4]).statistic
                

                # Get the mean correlation with all neighbouring channels
                corr[corr==0] = np.nan
                
                mean_corr.append(np.nanmean(corr))

    elif probe_type == 'Ad-tech':   
        for iGroup in range(int(data.shape[0]/8)): 
        # Select data for the tetrode of interest
            data_group = data[iGroup*8:iGroup*8+8]


            # For each channel in the tetrode of interest
            for i in range(data_group.shape[0]):
                corr = np.zeros(data_group.shape[0])
                for j in range(data_group.shape[0]):
                    if i!=j:
                        # Compute the pearson correlation between channel i and j
                        corr[j] = stats.pearsonr(data[i+iGroup*8],data[j+iTetrode*8]).statistic
                

                # Get the mean correlation with all neighbouring channels
                corr[corr==0] = np.nan
                
                mean_corr.append(np.nanmean(corr))    

    #open the excel table
    tbl = pd.read_excel(pathtbl)

    #creat a datframe with electrode names and correlation 
    cor_df = pd.DataFrame()
    cor_df['electrodes'] = chRegions
    cor_df['tetrode_cor'] = mean_corr

    #write the correlation in the table
    for i in range (0, len(cor_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == cor_df.iloc[i, 0]), 'tetrode_cor'] = cor_df.iloc[i, 1]
    
    #save table
    tbl.to_excel(pathtbl, index=False)

    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})

    
    plt.figure(figsize=(20,5))
    plt.title('Correlation coefficient with neighbouring channels')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,mean_corr)

    if save==1:
        plt.savefig(path+'Correlation_Coefficient_AllMicroChannels.jpg')



def variance_normalized(data,chRegions,path,pathtbl,sub,sess,probe_type,save=1):
    '''
    Compute the variance of each channel normalized by the mean variance of all neighbouring channels on the same tetrode
    If variance is high then it means that the channel is not recording the same activity than the ones around, so maybe it
    is broken or there is a problem with the plugging on your recording system


    Parameters
    ---------------------------
    data: array
        Matrix with your data with shape nChannels x nSamples

    chRegions: array of strings
        Vector with the name of your channels

    path: string
        Path where you want to save output figure
    
    pathtbl : string
        Path where the excel file of this patient is stored
    
    sub : string
        String containing the number or the ID of the patient in your database
    
    sess : string
        String containing the name of the session you are analyzing
    
    probe_type : string
        String containing the model of micro-electrodes you have in your dataset (for now: Dixi or Ad-tech only)



    save: Default=1 to save figure 

    Returns
    ---------------------------
    Matplotlib plot with values of variance normalized
    Values are also stored in the excel file


    see Tuyisenge, V., Trebaul, L., Bhattacharjee, M., Chanteloup-Forêt, B., Saubat-Guigui, C., Mîndruţă, I., Rheims, S., 
    Maillard, L., Kahane, P., Taussig, D., & David, O. (2018). 
    Automatic bad channel detection in intracranial electroencephalographic recordings using ensemble machine learning. 
    Clinical Neurophysiology, 129(3), 548‑554. https://doi.org/10.1016/j.clinph.2017.12.013


    '''
       
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)



    variance = list()

    if probe_type == 'Dixi':
        # Loop over all tetrodes
        for iTetrode in range(int(data.shape[0]/4)):

            # Select data of interest
            data_tetrode = data[iTetrode*4:iTetrode*4+4]


            # Loop over all channels in the tetrode of interest
            for i in range(data_tetrode.shape[0]):

                # Get the variance of channel i
                var_i = np.var(data_tetrode[i])
                var_j = np.zeros(data_tetrode.shape[0])
                for j in range(data_tetrode.shape[0]):
                    if i!=j:

                        # Get the variance of neighbouring channels
                        var_j[j] = np.var(data_tetrode[j])
                

                # Get the mean variance of neighbouring electrodes
                var_j[var_j==0] = np.nan
                
                mean_var_j = np.nanmean(var_j)


                # Normalized variance of channels i by mean variance of neighbouring channels
                variance.append(var_i/mean_var_j)

    elif probe_type == 'Ad-tech':
        # Loop over all tetrodes
        for iGroup in range(int(data.shape[0]/8)):

            # Select data of interest
            data_group = data[iGroup*8:iGroup*8+8]


            # Loop over all channels in the tetrode of interest
            for i in range(data_group.shape[0]):

                # Get the variance of channel i
                var_i = np.var(data_group[i])
                var_j = np.zeros(data_group.shape[0])
                for j in range(data_group.shape[0]):
                    if i!=j:

                        # Get the variance of neighbouring channels
                        var_j[j] = np.var(data_group[j])
                

                # Get the mean variance of neighbouring electrodes
                var_j[var_j==0] = np.nan
                
                mean_var_j = np.nanmean(var_j)


                # Normalized variance of channels i by mean variance of neighbouring channels
                variance.append(var_i/mean_var_j)



      #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    var_df = pd.DataFrame()
    var_df['electrodes'] = chRegions
    var_df['var'] = variance

    #write the correlation in the table
    for i in range (0, len(var_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == var_df.iloc[i, 0]), 'variance_norm'] = var_df.iloc[i, 1]
    
    #save table
    tbl.to_excel(pathtbl, index=False)



    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))

    plt.title('Variance normalized by neighbouring mico-channels')

    plt.xticks(rotation = 45)
    plt.plot(chRegions,variance)  

    if save==1:
        plt.savefig(path+'Variance_Normalized_AllChannels.jpg')



def deviation(data,chRegions,path,pathtbl,sub,sess,probe_type, save=1):
    '''
    Compute the deviation (i.e. electrical drift)

    Parameters
    ---------------------------
    data: array 
        Matrix with your data with shape nChannels x nSamples

    chRegions: array of strings
        Vector with the name of your channels

    path: string
        Path where you want to save the output's figure
    
    paththbl : string
        Path where to store values in excel

    sub : string
        IDs of the patient
    
    sess : string
        Name of the session
    
    probe_type : string
        String containing the model of micro-electrodes you have in your dataset (for now: Dixi or Ad-tech only)

    
    Returns
    ---------------------------
    Matplotlib plot and values in excel file



    see Tuyisenge, V., Trebaul, L., Bhattacharjee, M., Chanteloup-Forêt, B., Saubat-Guigui, C., Mîndruţă, I., Rheims, S., 
    Maillard, L., Kahane, P., Taussig, D., & David, O. (2018). 
    Automatic bad channel detection in intracranial electroencephalographic recordings using ensemble machine learning. 
    Clinical Neurophysiology, 129(3), 548‑554. https://doi.org/10.1016/j.clinph.2017.12.013
    '''



    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)



    deviation = list()

    if probe_type == 'Dixi':
        # Loop over all tetrodes
        for iTetrode in range(int(data.shape[0]/4)):

            # Select data of interest

            data_tetrode = data[iTetrode*4:iTetrode*4+4]



            # Loop over all channels in the tetrode of interest
            for i in range(data_tetrode.shape[0]):

                # Get the mean amplitude of channel i
                mean_i = np.mean(data_tetrode[i])
                mean_j = np.zeros(data_tetrode.shape[0])
                for j in range(data_tetrode.shape[0]):
                    if i!=j:

                        # Get the mean amplitude of neighbouring channels
                        mean_j[j] = np.mean(data_tetrode[j])
                

                # Get the mean of neighbouring electrodes' amplitudes 
                mean_j[mean_j==0] = np.nan
                
                mean_neighbours = np.nanmean(mean_j)


                # Get the deviation by substracting the mean of neighbours to channel i
                deviation.append(mean_i - mean_neighbours)

    elif probe_type == 'Ad-tech':
        # Loop over all tetrodes
        for iGroup in range(int(data.shape[0]/8)):

            # Select data of interest

            data_group = data[iGroup*8:iGroup*8+8]



            # Loop over all channels in the tetrode of interest
            for i in range(data_group.shape[0]):

                # Get the mean amplitude of channel i
                mean_i = np.mean(data_group[i])
                mean_j = np.zeros(data_group.shape[0])
                for j in range(data_group.shape[0]):
                    if i!=j:

                        # Get the mean amplitude of neighbouring channels
                        mean_j[j] = np.mean(data_group[j])
                

                # Get the mean of neighbouring electrodes' amplitudes 
                mean_j[mean_j==0] = np.nan
                
                mean_neighbours = np.nanmean(mean_j)


                # Get the deviation by substracting the mean of neighbours to channel i
                deviation.append(mean_i - mean_neighbours)
            

    # Z-score transformation
    deviation = stats.zscore(deviation)

    #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    dev_df = pd.DataFrame()
    dev_df['electrodes'] = chRegions
    dev_df['devi'] = deviation

    #write the correlation in the table
    for i in range (0, len(dev_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == dev_df.iloc[i, 0]), 'deviation'] = dev_df.iloc[i, 1]
  
    #save table
    tbl.to_excel(pathtbl, index=False)


    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))
    plt.title('Deviation Z-Score')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,deviation)  

    if save==1:
        plt.savefig(path+'Deviation_Zscore_AllChannels.jpg')



def variance(data,chRegions,path,pathtbl,sub,sess,save=1):
    '''
    Compute the variance of each channel to see if there is a lot of artefacts or not

    Parameters
    ---------------------------
    data: array
        Matrix with your data with shape nChannels x nSamples

    chRegions: array of strings
        Vector with the name of your channels

    path: string
        Path where you want to save output figure
    
    paththbl : string
        Path to the excel file where to store informations
    
    sub : string
        IDs of the patient
    
    sess : string
        Name of the session

    save: Default=1 to save figure 

    
    Returns 
    ---------------------------
    Matplotlib plot and value store in excel file

    '''
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)



    variance = list()

    # Loop over all tetrodes
    for iChannels in range(int(data.shape[0])):

        # Select data of interest
        data_channel = data[iChannels]

        # Get the variance of channel i
        var_i = np.var(data_channel)

        # Normalized variance of channels i by mean variance of neighbouring channels
        variance.append(var_i)
            
    #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    var_df = pd.DataFrame()
    var_df['electrodes'] = chRegions
    var_df['var'] = variance

    #write the correlation in the table
    for i in range (0, len(var_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == var_df.iloc[i, 0]), 'variance'] = var_df.iloc[i, 1]
  
    #save table
    tbl.to_excel(pathtbl, index=False)



    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))
    plt.title('Variance')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,variance)  

    if save==1:
        plt.savefig(path+'Variance_AllChannels.jpg')


def signaltonoise(a,chRegions,path,pathtbl,sub,sess, save=1, axis=1, ddof=0):
    """
    The signal-to-noise ratio of the input data.

    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.

    Parameters
    ---------------------------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.

    Returns
    ---------------------------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.

    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)

    
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    s2n = np.where(sd == 0, 0, m/sd)

    #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    var_df = pd.DataFrame()
    var_df['electrodes'] = chRegions
    var_df['SNR'] = s2n

    #write the correlation in the table
    for i in range (0, len(var_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == var_df.iloc[i, 0]), 'SNR'] = var_df.iloc[i, 1]
  
    #save table
    tbl.to_excel(pathtbl, index=False)


    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))
    plt.title('SNR')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,s2n)  

    if save==1:
        plt.savefig(path+'SNR_AllChannels.jpg')

    return s2n

def kurtosis(data,chRegions,path,pathtbl,sub,sess,save=1):
    """
    Kurtosis: An electrical activity may appear in one of the channels and be absent in the remaining ones. 
    Such events can be detected by computing the kurtosis in all channels. Given that the kurtosis 
    indicates the presence of outliers in datasets, the highest value reveals which channel shows a 
    particular event (Mognon et al., 2011) (from Tuyisenge et al., 2018)

    Parameters
    ---------------------------
    data : array
        Matrix containing your data with shape nChannels x nSamples
    
    chRegions : array of strings
        Vector containing name of your channels
    
    path : string
        String of the path where you want to store figures
    
    pathtbl : string
        Path where the excel file is stored
    
    sub : string
        IDs of the patient you are analyzing
    
    sess : string
        Name of the session 
    
    save : Boolean, default=1
        =1 if you want to save plot, put = 0 otherwise
    
    
    Results
    ---------------------------
    Matplotlib plot and value store in the excel file
    
    """

    kurtosis_channel = []

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)

    
    for iCh in range(data.shape[0]):
        a = data[iCh]

        # Compute kurtosis on each channel
        kurtosis_channel.append(stats.kurtosis(a,axis=0,fisher=True))
    
    
    #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    kur_df = pd.DataFrame()
    kur_df['electrodes'] = chRegions
    kur_df['kurt'] = kurtosis_channel
    #write the correlation in the table
    for i in range (0, len(kur_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == kur_df.iloc[i, 0]), 'kurtosis'] = kur_df.iloc[i, 1]
  
    #save table
    tbl.to_excel(pathtbl, index=False)


    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))
    plt.title('Kurtosis on all channels')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,kurtosis_channel)  

    if save==1:
        plt.savefig(path+'Kurtosis_AllChannels.jpg')




def hurst_component(data,chRegions,path,pathtbl,sub,sess,save=1):
    """
    Compute the Hurst component
    You can see Tuyisenge et al. (2018) for a detail of the algorithm
    Typically EEG data have values around 0.7


    Parameters
    ---------------------------
    data : array
        Matrix containing your data with shape nChannels x nSamples

    chRegions : array of strings
        Vectors containing the name of your channels
    
    path : string
        Path where you want to store the plots
    
    pathtbl: string
        Path where is store your excel file saving results

    sub: String
        String "sub-XX" where XX is the number of the subject analyzed
    
    sess: String
        Name of the session you are analyzing
    
    Returns
    ---------------------------
    Matplotlib plot
    """

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)
    hurst = list() 


    # Loop over each channel
    for iCh in range(data.shape[0]):

        # Step 1. Compute mean amplitude
        mean_amplitude = np.mean(data[iCh])

        # Step 2. Create a mean centered channel
        mean_center_channel = data[iCh] - mean_amplitude


        # Step 3. Compute the cumulative channel deviation
        chan_deviation = 0
        chan_deviation = [chan_deviation + mean_center_channel[t] for t in range(len(mean_center_channel))]


        # Step 4. Compute the channel amplitude range 
        amplitude_range = np.max(chan_deviation) - np.min(chan_deviation)


        # Step 5. Compute the standard deviation
        std_channel = np.std(data[iCh])

        # Step 6. Compute Hurst exponent
        hurst.append(np.sqrt(np.log(amplitude_range/std_channel)))



    #open the excel table
    tbl = pd.read_excel(pathtbl)


    #creat a datframe with electrode names and
    hurst_df = pd.DataFrame()
    hurst_df['electrodes'] = chRegions
    hurst_df['Hurst'] = hurst
    #write the correlation in the table
    for i in range (0, len(hurst_df)):   
        tbl.loc[(tbl['sub'] == sub) & (tbl['run'] == sess) & (tbl['electrodes'] == hurst_df.iloc[i, 0]), 'Hurst'] = hurst_df.iloc[i, 1]
  
    #save table
    tbl.to_excel(pathtbl, index=False)
    # Plot the results
    matplotlib.rcParams.update({'font.size': 11})


    plt.figure(figsize=(20,5))
    plt.title('Hurst exponent (in EEG ~ 0.7)')
    plt.xticks(rotation = 45)
    plt.plot(chRegions,hurst)  

    if save==1:
        plt.savefig(path+'Hurst_Exponent_AllChannels.jpg')


