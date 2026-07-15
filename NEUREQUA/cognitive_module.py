# Import librairies

from __future__ import division

import os
import warnings
import numpy as np
import datetime
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sb
import os
import mne
import scipy.stats as stats
import json




# coding=utf-8

'''
This function included in NeuReQua is from @alafuzof

https://github.com/alafuzof/NeuralynxIO
'''

HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header

NCS_SAMPLES_PER_RECORD = 512
NCS_RECORD = np.dtype([('TimeStamp',       np.uint64),       # Cheetah timestamp for this record. This corresponds to
                                                             # the sample time for the first data point in the Samples
                                                             # array. This value is in microseconds.
                       ('ChannelNumber',   np.uint32),       # The channel number for this record. This is NOT the A/D
                                                             # channel number
                       ('SampleFreq',      np.uint32),       # The sampling frequency (Hz) for the data stored in the
                                                             # Samples Field in this record
                       ('NumValidSamples', np.uint32),       # Number of values in Samples containing valid data
                       ('Samples',         np.int16, NCS_SAMPLES_PER_RECORD)])  # Data points for this record. Cheetah
                                                                                # currently supports 512 data points per
                                                                                # record. At this time, the Samples
                                                                                # array is a [512] array.

NEV_RECORD = np.dtype([('stx',           np.int16),      # Reserved
                       ('pkt_id',        np.int16),      # ID for the originating system of this packet
                       ('pkt_data_size', np.int16),      # This value should always be two (2)
                       ('TimeStamp',     np.uint64),     # Cheetah timestamp for this record. This value is in
                                                         # microseconds.
                       ('event_id',      np.int16),      # ID value for this event
                       ('ttl',           np.int16),      # Decimal TTL value read from the TTL input port
                       ('crc',           np.int16),      # Record CRC check from Cheetah. Not used in consumer
                                                         # applications.
                       ('dummy1',        np.int16),      # Reserved
                       ('dummy2',        np.int16),      # Reserved
                       ('Extra',         np.int32, 8),   # Extra bit values for this event. This array has a fixed
                                                         # length of eight (8)
                       ('EventString',   'S', 128)])  # Event string associated with this event record. This string
                                                         # consists of 127 characters plus the required null termination
                                                         # character. If the string is less than 127 characters, the
                                                         # remainder of the characters will be null.

VOLT_SCALING = (1, u'V')
MILLIVOLT_SCALING = (1000, u'mV')
MICROVOLT_SCALING = (1000000, u'µV')

#############
### UTILS ###
#############

def ensure_dir(path: str) -> None:
    """
    Create the directory if it does not exist


    Parameters
    ----------
    path: str
        Path where you want to create folder
    """

    # Check if the path exists
    isExist = os.path.exists(path)

    # It is doesn't create it
    if not isExist:
        os.makedirs(path)



def find_nearest(array, value):
    '''
    Find the index in the array closest to a desired value.

    Parameters
    ----------
    array: Numpy array
        A 1-D array containing numerical values
    
    value: float
        Value you want to find in array

    Returns
    ----------
    idx: int
        Index in the array closest to value
    '''
    # Make sure that array is an array and not a list
    array = np.asarray(array)

    # Find where the difference between value and elements of array is min
    idx = (np.abs(array - value)).argmin()

    # Returns
    return idx

def read_header(fid):
    '''
    Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
    
    Parameters
    ----------
    fid: file object
        File object to .ncs file recorded with Neuralynx

    Returns
    -------
    raw_hdr: string
        Informations about the recording extracted from the Neuralynx header
    '''
    # Get the current position in the file stream
    pos = fid.tell()

    # Set reference point to beginning of file
    fid.seek(0)

    # Read the header
    raw_hdr = fid.read(HEADER_LENGTH).strip(b'\0')
    fid.seek(pos)

    return raw_hdr


def parse_header(raw_hdr):
    '''
    Parse the header string into a dictionnary of name value pairs

    Parameters
    ----------
    raw_hdr: string
        Informations about the recording extracted from the Neuralynx header (obtained with read_header)

    Returns
    -------
    hdr: dict
        Informations about the recording but stored in a dictionnary
    '''
    # Parse the header string into a dictionary of name value pairs
    hdr = dict()

    # Decode the header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
    raw_hdr = raw_hdr.decode('iso-8859-1')

    # Neuralynx headers seem to start with a line identifying the file, so
    # let's check for it
    hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
    if hdr_lines[0] != '######## Neuralynx Data File Header':
        warnings.warn('Unexpected start to header: ' + hdr_lines[0])

    # Try to read the original file path
    try:
        assert hdr_lines[1].split()[1:3] == ['File', 'Name']
        hdr[u'FileName']  = ' '.join(hdr_lines[1].split()[3:])
        # hdr['save_path'] = hdr['FileName']
    except:
        warnings.warn('Unable to parse original file path from Neuralynx header: ' + hdr_lines[1])

    # Process lines with file opening and closing times
    hdr[u'TimeOpened'] = hdr_lines[2][3:]
    hdr[u'TimeOpened_dt'] = parse_neuralynx_time_string(hdr_lines[2])
    hdr[u'TimeClosed'] = hdr_lines[3][3:]
    hdr[u'TimeClosed_dt'] = parse_neuralynx_time_string(hdr_lines[3])

    # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
    for line in hdr_lines[4:]:
        try:
            name, value = line[1:].split()  # Ignore the dash and split PARAM_NAME and PARAM_VALUE
            hdr[name] = value
        except:
            warnings.warn('Unable to parse parameter line from Neuralynx header: ' + line)

    return hdr


def read_records(fid, record_dtype, record_skip=0, count=None):
    '''
    Read count records (default all) from the file object fid skipping the first record_skip records. 
    Restores the position of the file object after reading.

    When multiple recordings segment in the .ncs file

    Parameters
    ----------
    fid: file object
        File object of the recording files
    
    record_dtype: np.dtype
        Data type of all objects in the .nev file
    
    record_skip: int (Default=0)
        Record object to skip, if zero it means we include all recordings in the files
        If = 1 then skip the first one
    
    count: int (Default=None)
        Number of items to read. If None then means -1 and means all items

    Returns
    -------
    rec: np.array
        Numpy array extracted from data in text or binary file 
        see https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
    '''
    # Read count records (default all) from the file object fid skipping the first record_skip records. Restores the
    # position of the file object after reading.
    if count is None:
        count = -1

    pos = fid.tell()
    fid.seek(HEADER_LENGTH, 0)
    fid.seek(record_skip * record_dtype.itemsize, 1)
    rec = np.fromfile(fid, record_dtype, count=count)
    fid.seek(pos)

    return rec





def parse_neuralynx_time_string(time_string):
    '''
    Parse a datetime object from the idiosyncratic time string Neuralynx file headers

    Parameters
    ----------
    time_string: string
        String containing time from Neuralynx file headers

    Returns
    -------
    datetime.datetime: datetime object
        A datetime object is a single object containing all the information from a date object and a time object.
        see https://docs.python.org/3/library/datetime.html#datetime-objects
    '''
    # Parse a datetime object from the idiosyncratic time string in Neuralynx file headers
    try:
        tmp_date = [int(x) for x in time_string.split()[4].split('/')]
        tmp_time = [int(x) for x in time_string.split()[-1].replace('.', ':').split(':')]
        tmp_microsecond = tmp_time[3] * 1000
    except:
        warnings.warn('Unable to parse time string from Neuralynx header: ' + time_string)
        return None
    else:
        return datetime.datetime(tmp_date[2], tmp_date[0], tmp_date[1],  # Year, month, day
                                 tmp_time[0], tmp_time[1], tmp_time[2],  # Hour, minute, second
                                 tmp_microsecond)








def load_nev(file_path):
    '''
    Load Events.nev file from Neuralynx acquisition system

    Parameters
    ----------
    file_path: string or path-like
        Path where the file .nev is stored

    Returns
    -------
    nev: dict
        Dictionnary containing events informations (as TimeStamp, id, ttl values)
    '''
    # Load the given file as a Neuralynx .nev event file and extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NEV_RECORD)

    header = parse_header(raw_header)

    # Check for the packet data size, which should be two. DISABLED because these seem to be set to 0 in our files.
    #assert np.all(record['pkt_data_size'] == 2), 'Some packets have invalid data size'


    # Pack the extracted data in a dictionary that is passed out of the function
    nev = dict()
    nev['file_path'] = file_path
    nev['raw_header'] = raw_header
    nev['header'] = header
    nev['records'] = records
    nev['events'] = records[['pkt_id', 'TimeStamp', 'event_id', 'ttl', 'Extra', 'EventString']]

    return nev


def to_json(dictionary, filename):
    '''
    Save a dictionnary to a .json file on your disk.

    Parameters
    ----------
    dictionary: dict
        The dictionnary you want to save
    
    filename: str or path-like
        Path and name of the .json file you want to create with the informations
        contained in dictionary
    '''
    with open(filename,'w') as fp:
        json.dump(dictionary, fp,sort_keys=True, indent=4,ensure_ascii=False)
 


def ensure_raw(path,sub,sess):
    '''
    Look if data were already created on the disk or not (.npy file).
    If they were created then load them as a memory mapped object.

    Parameters
    ----------
    path: string or path-like
        Path of the folder where your data are stored
    
    sub: str
        Id of the patient to analyze according to your dataset

    sess: str
        Name of the experimental session to analyze
    
    Returns
    -------
    lfps: np.memmap
        Memory mapped object of the raw data stored on your file (.npy file)
    '''
    # Try to load the raw data
    try:
        lfps = np.squeeze(np.lib.format.open_memmap(path+'/raw_data_'+sub+'_'+sess+'.npy',mode='r+',dtype=np.int16))

        return lfps
    
    except:
        raise FileNotFoundError(
            f"Raw data do not exist, launch load_neuralynx_micro first"
        )
    


def load_neuralynx_micro(path, sub, sess, macro_pattern='_sub',verbose=True):
    """
    Load intracranial EEG microwire recordings acquired with Neuralynx
    in .ncs format into memory, excluding macrocontact channels.
 
    The function reads all .ncs files located in the specified directory,
    automatically excludes channels whose filename contains the macro-
    contact pattern, loads the full recording into memory, and returns
    both a structured MNE Raw object and a raw NumPy array.
 
    Parameters
    ----------
    path : str
        Path to the directory containing the .ncs recording files.
 
    macro_pattern : str, optional
        Substring used to identify and exclude macrocontact channels
        from the recording (default: '_sub').
        Example: if macrocontact files are named 'LA1_sub.ncs',
        setting macro_pattern='_sub' will exclude them automatically.
 
    verbose : bool, optional
        If True, prints a summary of the loaded recording upon
        completion (default: True).
 
    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing the full microwire recording.
        Provides access to channel metadata (names, sampling rate,
        recording duration) and supports MNE-based preprocessing.
 
    data : np.ndarray
        NumPy array of shape (n_channels, n_samples) containing the
        raw signal values in Volts, loaded entirely into memory.
 
    metadata : dict
        Dictionary containing key recording parameters:
            - 'ch_names'    : list of str, microwire channel labels
            - 'n_channels'  : int, number of microwire channels
            - 'sfreq'       : float, sampling rate in Hz
            - 'duration_s'  : float, recording duration in seconds
            - 'n_samples'   : int, total number of samples per channel
            - 'macro_pattern': str, exclusion pattern used
 
    Raises
    ------
    FileNotFoundError
        If the specified path does not exist or contains no .ncs files.
 
    ValueError
        If no microwire channels remain after applying the exclusion
        pattern (i.e., all channels matched the macro_pattern).

    """
 
    # ------------------------------------------------------------------ #
    # 1. Validate input path
    # ------------------------------------------------------------------ #
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The specified path does not exist: '{path}'"
        )
 
    ncs_files = [f for f in os.listdir(path) if f.endswith('.ncs')]
    if len(ncs_files) == 0:
        raise FileNotFoundError(
            f"No .ncs files found in: '{path}'"
        )
 
    if verbose:
        print(f"Found {len(ncs_files)} .ncs file(s) in '{path}'")
        print(f"Excluding channels matching pattern: '*{macro_pattern}'")
 
    # ------------------------------------------------------------------ #
    # 2. Load microwire channels (exclude macrocontacts)
    # ------------------------------------------------------------------ #
    raw = mne.io.read_raw_neuralynx(
        path,
        exclude_fname_patterns=[f'*{macro_pattern}'],
        preload=False,    # lazy load first to inspect metadata
        verbose=False
    )
 
    if len(raw.ch_names) == 0:
        raise ValueError(
            f"No microwire channels remain after excluding pattern "
            f"'*{macro_pattern}'. "
            f"Check that macro_pattern matches your macrocontact "
            f"filename suffix."
        )
 
    if verbose:
        n_excluded = len(ncs_files) - len(raw.ch_names)
        print(f"  → {n_excluded} macrocontact channel(s) excluded")
        print(f"  → {len(raw.ch_names)} microwire channel(s) retained")
 
    # ------------------------------------------------------------------ #
    # 3. Load full recording into memory
    # ------------------------------------------------------------------ #
    from pathlib import Path

    my_file = Path(path+'raw_data_'+sub+'_'+sess+'.npy')
    if my_file.is_file():
        data = np.squeeze(np.lib.format.open_memmap(path+'/raw_data_'+sub+'_'+sess+'.npy',mode='r+',dtype=np.int16))
        print("File already exist")
    else:

        if verbose:
            print("Loading full recording into memory...")
        
        data1 = list()

        for iChannels in range(len(raw.ch_names)):

            # Load data for the channel of interest
            data_ch = raw.get_data(picks=iChannels)

            # Transform into micro-volts
            data_microvolt = data_ch * 10**6

            # Transform into int16 to save space
            data2save = data_microvolt.astype(np.int16)

            # Store into the list
            data1.append(data2save)
        

        data = np.array(data1)

        del data1

        np.save(path+'raw_data_'+sub+'_'+sess+'.npy',data)

 
    # ------------------------------------------------------------------ #
    # 4. Build metadata dictionary
    # ------------------------------------------------------------------ #
    metadata = {
        'ch_names'     : raw.ch_names,
        'n_channels'   : len(raw.ch_names),
        'sfreq'        : raw.info['sfreq'],
        'duration_s'   : raw.times[-1],
        'n_samples'    : data.shape[1],
        'macro_pattern': macro_pattern
    }
 
    # ------------------------------------------------------------------ #
    # 5. Print summary
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n── Recording summary ──────────────────────────────")
        print(f"  Channels   : {metadata['n_channels']}")
        print(f"  Samp. rate : {metadata['sfreq']:.0f} Hz")
        print(f"  Duration   : {metadata['duration_s']:.1f} s "
              f"({metadata['duration_s']/60:.2f} min)")
        print(f"  Samples    : {metadata['n_samples']:,}")
        print(f"  Data shape : {data.shape}  (channels × samples)")
        print(f"  Channel labels:")
        for ch in metadata['ch_names']:
            print(f"    • {ch}")
        print("───────────────────────────────────────────────────\n")

    
    
        


    # Save dictionnary to disk
    to_json(metadata,path+'metadata.json')
        

 
    return raw, metadata



#####################
### Preprocessing ###
#####################

def create_epoch(lfps,Folder,t_min=1,t_max=1,ds_factor=1):
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

    Returns
    -------
    epoch_data: np.array
        Array containing the data of each epoch with 3-D shape (nEpoch x nChannels x nSamples)
    """
    
    
    lfps = np.squeeze(lfps)

    # Load the events.nev
    nev = load_nev(Folder+'./Events.nev')  # Load event data into a dictionary


    # Only keep events that are not fixation cross
    idx_trials = np.nonzero(nev['events']['ttl'])



    # Get the timestamps of each trials
    ts_trials = nev['events']['TimeStamp'][idx_trials]


    # Get the timestamps relative to the onset of the recording
    ts_relatif = ts_trials - nev['events']['TimeStamp'][0]


    # Timestamps are expressed in micro-seconds in Neuralynx so divide by 10^6
    ts_second = ts_relatif / 10**6


    # Transform into samples
    onset_sample = ts_second*32768

    # Transfrom into int (because samples are index)
    onset_sample = onset_sample.astype(int)

    # Initialize list to store each epoch
    epoch_data = list()

    # Loop over all events 
    for iEvent in range(len(onset_sample)):

        epoch = lfps[:,onset_sample[iEvent]-int(t_min*32768):onset_sample[iEvent]+int(t_max*32768)]

        epoch_data.append(epoch)
    


    return np.array(epoch_data)


def filt_butter(data, lowcut=300, highcut=3000, btype='bandpass', sr=32768, order=2):
    """
    Apply a butterworth band-pass filter

    Parameters
    ---------------------------
    data: ND-array
        A 1-D array containing your LFPs activity

    lowcut: int
        Lower frequency of your bandpass filter
    
    highcut: int
        Higher frequency of your bandpass filter

    btype: string, default = 'bandpass'
        Type of filter to use.
        Can be either 'bandpass' (default), 'lowpass' or 'highpass'.

    sr: int; default = 32768
        Sampling frequency of your recording system
    
    order: int, default = 2
        Order of the butterworth filter
    
    Returns
    ---------------------------
    filt_data: ND-array
        1-D array containing the LFPs filter between frequencies specified
    """
    from scipy.signal import butter, sosfilt
    
    def butter_(lowcut, highcut, btype, sr, order=order):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        if btype == 'highpass':
            sos = butter(order, high, btype=btype, output='sos')
        elif btype == 'lowpass':
            sos = butter(order, low, btype=btype, output='sos')
        elif btype == 'bandpass':
            sos = butter(order, [low, high], btype=btype, output='sos')
        return sos
    
    sos = butter_(lowcut, highcut, btype, sr, order=order)
    filt_data = sosfilt(sos, data)
    
    return filt_data





###################################    
### Cognitive module - Plotting ###
###################################

def plot_artefact_map(path,sub,sess):
    """
    Plot figure to show the variance of each trial and each channel
    Enables us to quickly see the channels that are artefacted (e.g., by epileptic activities)
    and also trials contaminated

    Just like the figure 9.B of Mercier et al. (2022)

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
    ensure_dir(path)


    lfps = ensure_raw(path,sub,sess)
    
    # Then create epoch
    epoch_data = create_epoch(lfps,path,t_min=1, t_max=1)
    
    var_ch = list()

    nCh = epoch_data.shape[1]

    for iCh in range(epoch_data.shape[1]):
        data_channel = epoch_data[:,iCh,:]

        # Compute the variance for each trial
        variance_trial = np.var(data_channel,1)
    
        var_ch.append(variance_trial)
   

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(12, 4),layout='constrained')
    grid = plt.GridSpec(4,4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, :3])
    y_hist = fig.add_subplot(grid[:-1:, 3:], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, :3], yticklabels=[], sharex=main_ax)

    # scatter points on the main axes
    sb.heatmap(var_ch,ax=main_ax,cbar=False,cmap="rocket_r") # pour l'instant rocket_r est la mieux
    main_ax.axes.get_xaxis().set_visible(False)
    main_ax.locator_params(axis='y',nbins=int(nCh/4+1)) 
    main_ax.set_ylabel('# Channels',size=15)

    # histogram on the attached axes
    x_hist.plot(np.mean(var_ch,0),'.',color='coral')
    # Setting the number of ticks 
    x_hist.locator_params(axis='x',nbins=10) 
    x_hist.set_xlabel('# Trials',loc='center',size=15)

    y = np.arange(len(var_ch))
    y_hist.plot(np.mean(var_ch,1),y,'.',color='coral')
    y_hist.axes.get_yaxis().set_visible(False)
    y_hist.set_title('# Channels',loc='center',size=15)


    plt.savefig(path+'Artefact_Map.jpg',dpi=800)

    plt.show()

    plt.close()



 



def plot_erp(path,metadata,sub,sess,tmin,tmax,mua=True):
    '''
    Plot Event-related potentials in response to all events loaded from your recording.
    Plot the ERPs for each channel that you had. It also plot the MUA activity estimated firing rate.

    Parameters
    ----------
    path: str or path-like
        Path where the .npy file containing your raw data is stored

    metadata: dict
        Dictionnary created when loading your data containing informations about the recordings
    
    sub: str
        Id of the patient to analyze 
    
    sess: str
        Name of the experimental session to analyze

    tmin: float
        Time to include before events (must be positive)

    tmax: float
        Time to include after events (must be positive)

    mua: Boolean, default = True
        Either to plot or not the MUA activity alonged ERP
    '''

    # Check whether the specified path exists or not
    ensure_dir(path)

    # Extract the sampling frequency from metadata object
    sr = int(metadata['sfreq'])

    # Check if .npy file with data exists
    lfps = ensure_raw(path,sub,sess)
        
    # Then create epoch
    if mua:
        epoch_data = create_epoch(lfps,path,t_min=tmin+0.1, t_max=tmax+.1)
    else:
        epoch_data = create_epoch(lfps,path,t_min=tmin, t_max=tmax)

    # Get the mean activity across trials for each channels
    mean_channels = np.mean(epoch_data,axis=0)

    # Get error standard
    sem_channels = stats.sem(epoch_data,axis=0)

    # Create a time array to get time associated with each sample
    time = np.linspace(-tmin,tmax,epoch_data.shape[2])
    
    # Extract number of channels
    nChannels = mean_channels.shape[0]

    # Loop over each channel in your recording
    for iChannels in range(nChannels):

        # If you want to plot MUA with ERPs
        if mua:

            # Parameters for gaussian window
            N_s = 0.05 # 50 ms time window
            sigma_s = 0.01 # 10 ms std

            kernel = sig.windows.gaussian(int(N_s*sr),int(sigma_s*sr))

            # Automatic threshold to detect spikes (from Quiroga et al., 2004)
            threshold = 4 * (np.median((np.absolute(filt_butter(lfps[iChannels,:],sr=sr))/0.6745)))

            # Initialize lists
            spikes_smooth = list()
            spikes = np.zeros((epoch_data.shape[0],epoch_data.shape[2]))

            # Loop over all trials
            for iTrials in range(epoch_data.shape[0]):

                # Apply band-pass filter on epochs
                epoch_filt = filt_butter(epoch_data[iTrials,iChannels,:],sr=32768)

                # Detect where above the threshold
                tEvents = np.where(epoch_filt> threshold)

                # Replace zeros by ones
                spikes[iTrials,tEvents] = 1

                # Apply convolution with gaussian filter
                spikes_smooth.append(np.convolve(spikes[iTrials], kernel,mode='same'))

            # Adjuste to keep only between -tmin and tmax
            times = np.linspace(-tmin+0.1,tmax+0.1,epoch_data.shape[2])

            idx_debut = find_nearest(times,-tmin)
            idx_fin = find_nearest(times,tmax)

            time_ok = times[idx_debut:idx_fin]
            fr_ok = np.array(spikes_smooth)[:,idx_debut:idx_fin]
            
            # Get the mean firing rate across time
            mean_fr = np.mean(fr_ok,axis=0)
            # Get error standard
            sem_fr = stats.sem(fr_ok,axis=0)

            # Crate the figure for plotting
            fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,layout='constrained')
            

            # Plot mean lfps
            ax1.plot(time_ok,mean_channels[iChannels,idx_debut:idx_fin],color='black')

            # Add standard error around mean lfps
            ax1.fill_between(time_ok,mean_channels[iChannels,idx_debut:idx_fin]-sem_channels[iChannels,idx_debut:idx_fin],
                            mean_channels[iChannels,idx_debut:idx_fin]+sem_channels[iChannels,idx_debut:idx_fin],
                            color='slategrey',alpha=0.4)

            # Add a vertical line at the onset of stimuli
            ax1.vlines(0,ymin=ax1.get_ylim()[0],ymax=ax1.get_ylim()[1],
                    linewidth=3,linestyle='--',color='darkorange')
            
            # Axes titles and labels
            ax1.set_title(metadata['ch_names'][iChannels] + ' - ERPs')
            ax1.set_ylabel('LFPs - Microvolts')


            ax2.plot(time_ok,mean_fr,color='black')

            # Add standard error around mean lfps
            ax2.fill_between(time_ok,mean_fr-sem_fr,
                            mean_fr+sem_fr,
                            color='slategrey',alpha=0.4)

            ax2.vlines(0,ymin=ax2.get_ylim()[0],ymax=ax2.get_ylim()[1],
                    linewidth=3,linestyle='--',color='darkorange')
            
            ax2.set_title('MUA Activity')
            ax2.set_ylabel('Firing rate (Hz)')
            ax2.set_xlabel('Time (s)')
            

            # Path where to store the results 
            path2save = path+'ERPs/'

            # Make sure it exists
            ensure_dir(path2save)

            # Save the plot with ERP
            plt.savefig(path2save+'ERP_MUA_'+metadata['ch_names'][iChannels]+'.jpg')
            
            # Close figure object
            plt.close()


        else:


            # Crate the figure for plotting
            fig, ax = plt.subplots(1,1,layout='constrained')
            

            # Plot mean lfps
            ax.plot(time,mean_channels[iChannels],color='black')

            # Add standard error around mean lfps
            ax.fill_between(time,mean_channels[iChannels]-sem_channels[iChannels],
                            mean_channels[iChannels]+sem_channels[iChannels],
                            color='slategrey',alpha=0.4)

            # Add a vertical line at the onset of stimuli
            ax.vlines(0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],
                    linewidth=3,linestyle='--',color='darkorange')
            
            # Axes titles and labels
            ax.set_title(metadata['ch_names'][iChannels])
            ax.set_ylabel('LFPs - Microvolts')
            ax.set_xlabel('Time (s)')

            # Path where to store the results 
            path2save = path+'ERPs/'

            # Make sure it exists
            ensure_dir(path2save)

            # Save the plot with ERP
            plt.savefig(path2save+'ERP_'+metadata['ch_names'][iChannels]+'.jpg')
            
            # Close figure object
            plt.close()
        





