import numpy as np
import scipy
import scipy.stats as stats
import seaborn as sb

from scipy.signal import filtfilt, bessel
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ EXPLANATION -----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
"""
Explanation of Inputs Needed to Run the Artifact Detection Pipeline
This script performs artifact detection in three steps:

Detect movement artifacts: Large, abrupt changes in the signal caused by patient movement or external disturbances.
Detect subtle artifacts: More localized, high-frequency distortions that may affect signal quality.
Merge detected windows: Combine overlapping or adjacent detected artifacts into a final list of artifact windows.
1️⃣ Required Inputs and Their Definitions
Before running the script, you need to define the following inputs.

📌 EEG/LFP Data
chTraces (ndarray): A 2D NumPy array of shape (n_channels, n_timepoints) containing the EEG/LFP signals.
Each row represents a different recording channel.
Each column represents a time point in the recording.
📌 Channel and Tetrode Information
chGoodInds (list): A list of good channel indices that are free of recording failures or noise.
tetInds (list of lists): A list of tetrode channel groups.
Each element is a list of channel indices corresponding to one tetrode.
Used for refining artifact detection by checking signal correlations within tetrodes.
chRegs (list): A list of region names or channel assignments (e.g., ['Hippocampus', 'Cortex', 'Thalamus']).
Used for grouping channels in subtle artifact detection.
📌 Recording Parameters
nCh (int): Total number of channels in the recording.
SR (int): Sampling rate in Hz.
Default is 20000 Hz (high-resolution data).
This is needed to convert artifact windows from samples to milliseconds.
📌 Movement Artifact Detection Parameters
These parameters are used in detect_movement_artifacts():

thresh_mvA (float): Z-score threshold for movement artifact detection.
Default: 3 (higher values detect fewer artifacts).
extend_mvA_ms (int): Time extension for detected movement artifacts (in ms).
Default: 200 ms (extends the detected artifact period).
threshCorr (float): Correlation threshold between tetrode wires for refinement.
Default: 0.6 (low values allow more artifacts to be kept).
📌 Subtle Artifact Detection Parameters
These parameters are used in detect_subtle_artifacts():

Filtering parameters:
lowcut_detPeaks (int): Low cut-off frequency for peak detection (Hz). Default: 300 Hz.
highcut_detPeaks (int): High cut-off frequency for peak detection (Hz). Default: 7000 Hz.
order_detPeaks (int): Filter order for bandpass filtering. Default: 2.
Thresholds for artifact detection:
thresh_dt1 (float): Z-score threshold for first detector. Default: 10.
nContacts_dt1 (int): Minimum channels needed to confirm an artifact. Default: 3.
nTetrodes_dt1 (int): Minimum tetrodes required to confirm an artifact. Default: 2.
extend_final_ms_dt1 (int): Time extension of detected artifacts (ms). Default: 20 ms.
Peak detection (high-frequency bursts)
thresh_peakDetect (float): Amplitude threshold for peak detection. Default: 4.5.
window_ms_dt2 (int): Window size for detecting rapid spiking (ms). Default: 5 ms.
thresh_dt2 (int): Spike count threshold for second detector. Default: 10.
nContacts_dt2 (int): Minimum channels required to confirm a high-frequency burst artifact. Default: 2.
nTetrodes_dt2 (int): Minimum tetrodes required to confirm a burst. Default: 2.
extend_final_ms_dt2 (int): Time extension of detected bursts (ms). Default: 20 ms.
📌 Window Merging Parameters
These are used in get_timeWinsIntersect() to merge detected artifact windows:

extend_final_com_ms (int): Merging window threshold (ms).
If two artifact windows are closer than this value, they are merged into one.
Default: 200 ms.
2️⃣ What This Script Does
Runs detect_movement_artifacts() on chTraces to find large movement artifacts.
Runs detect_subtle_artifacts() to detect high-frequency distortions.
Uses get_timeWinsIntersect() to merge the detected artifact windows into a final set.

3️⃣ Example of Running the Code
python
Copy
Edit
# Step 1: Detect movement artifacts
badWins_movement = detect_movement_artifacts(
    chTraces, chGoodInds, tetInds, verbose=True, 
    SR=20000, thresh_mvA=3, extend_mvA_ms=200, threshCorr=0.6
)

# Step 2: Detect subtle artifacts
badWins_subtle = detect_subtle_artifacts(
    chTraces, chGoodInds, tetInds, chRegs, SR=20000, verbose=True,
    extend_final_com_ms=200, lowcut_detPeaks=300, highcut_detPeaks=7000, order_detPeaks=2, 
    thresh_dt1=10, nContacts_dt1=3, nTetrodes_dt1=2, extend_final_ms_dt1=20, 
    thresh_peakDetect=4.5, window_ms_dt2=5, thresh_dt2=10, nContacts_dt2=2, 
    nTetrodes_dt2=2, extend_final_ms_dt2=20
)

# Step 3: Merge detected windows using intersection
badWins_final = get_timeWinsIntersect(badWins_movement, badWins_subtle, chTraces.shape[1])

# Display results
print(f"Total movement artifact windows: {len(badWins_movement)}")
print(f"Total subtle artifact windows: {len(badWins_subtle)}")
print(f"Final merged artifact windows: {len(badWins_final)}")

4️⃣ Summary of What You Need to Define
Parameter	Type	Description
chTraces	ndarray	EEG/LFP data (channels x time)
chGoodInds	list	Indices of good channels
tetInds	list	List of tetrode channel groups
chRegs	list	List of region names per channel
SR	int	Sampling rate in Hz (default: 20000)
thresh_mvA	float	Z-score threshold for movement artifacts
extend_mvA_ms	int	Time extension for movement artifacts (ms)
threshCorr	float	Correlation threshold for refining movement artifacts
lowcut_detPeaks	int	Low cut-off for peak detection (Hz)
highcut_detPeaks	int	High cut-off for peak detection (Hz)
order_detPeaks	int	Filter order for bandpass filtering
thresh_dt1	float	Z-score threshold for first subtle artifact detector
extend_final_com_ms	int	Window merge threshold (ms)


Made by Adrien Causse
"""


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ HOUSEKEEPING ----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
        
    return ind

def bandpass_filter_bessel(data, lowcut, highcut, sr, order=2):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = bessel(N=order, Wn=[low, high], btype='bandpass', analog=False, output='ba')
    
    y = filtfilt(b, a, data)
    return y
    
def smooth_binary_array(binary_array, kernel_size):
    kernel = np.ones(kernel_size)
    return np.convolve(binary_array, kernel, mode="same") > 0
    
    
def get_peri_stimulus_counts(actVect, window_ms=10, sampling_rate=1250):
    """
    Compute peri-stimulus histogram for each spike in a spike train using vectorized operations.
    
    Parameters:
    - actVect (np.array): Binary vector (1 for spike, 0 for no spike).
    - window_ms (int): Window size in milliseconds around each spike.
    - sampling_rate (int): Sampling rate in Hz.

    Returns:
    - peri_stimulus_counts (np.array): Array of the same length as spkTimes
      with the sum of spikes in the ±window around each spike.
    """
    spkTimes = np.where(actVect)[0]  # Get spike times (indices)
    
    # Convert window from ms to samples
    window_samples = int(window_ms * sampling_rate / 1000)
    
    # Compute the number of spikes in the window for each spike using broadcasting
    peri_stimulus_counts = (
        np.sum(
            (spkTimes[:, None] - spkTimes[None, :]) <= window_samples, axis=1
        ) 
        - np.sum(
            (spkTimes[:, None] - spkTimes[None, :]) < -window_samples, axis=1
        ) 
        - 1  # Remove self-count
    )

    return peri_stimulus_counts

def get_timeWins4AdjacentPoints(arr):
    """
    return start_ends
    
    which is an array of shape (X, 2)
    arr in input is made of indices (np.where(bool))
    """
    arr = np.array(arr)
    # Find the indices where the difference between consecutive elements is not 1
    diff = np.diff(arr)
    breaks = np.where(diff != 1)[0]
    
    # The start of each burst is the element after each break, plus the first element
    starts = np.insert(arr[breaks + 1], 0, arr[0])
    
    # The end of each burst is the element at each break, plus the last element
    ends = np.append(arr[breaks], arr[-1])
    
    return np.vstack((starts, ends)).T


def get_timeWins_mergeIfNext(timeWins, nbPointsToMerge):
    """
    Merges overlapping or close windows where gaps are < nbPointsToMerge points, and returns the indices of the merged windows.

    Parameters:
    - timeWins (ndarray): Shape (n, 2), each row is [start, end].
    - nbPointsToMerge (int): Max gap allowed between consecutive windows to merge.

    Returns:
    - merged_windows (ndarray): Shape (m, 2), merged windows.
    - merged_indices (list of lists): Indices of the original windows contributing to each merged window.
    """
    if len(timeWins) == 0:
        return np.array([]).reshape(0, 2), []

    # Sort windows by start time, while keeping track of original indices
    sorted_indices = np.argsort(timeWins[:, 0])
    sorted_windows = timeWins[sorted_indices]

    merged = [sorted_windows[0].tolist()]  # Initialize with first window
    merged_indices = [[sorted_indices[0]]]  # Track indices of merged windows

    for idx, (start, end) in zip(sorted_indices[1:], sorted_windows[1:]):
        # Check if the current window should be merged
        if start - merged[-1][1] < nbPointsToMerge:  
            merged[-1][1] = max(merged[-1][1], end)  # Extend last window
            merged_indices[-1].append(idx)  # Add original index to merged group
        else:
            merged.append([start, end])  # Start a new window
            merged_indices.append([idx])  # Start new index group

    return np.array(merged), merged_indices


def get_timeWinsTemplatedSignal(signal, timeWins, nTimePoints = None, filling = np.nan):
    """
    return templatedSignal
    
    signal must be a 'pure' np.array, VECTOR shape. If it is an array of lists of len 1, please use: signal = np.ravel(signal) before passing signal into the function 
    timeWins is a np.array of shape (nWins, 2) / 2 for start and for stop
    """
    if nTimePoints is None:
        nTimePoints = signal.shape[0]
    templatedSignal = np.full(nTimePoints, filling)
    for timeWin in timeWins:
        templatedSignal[timeWin[0]:timeWin[1]] = signal[timeWin[0]:timeWin[1]]
    return templatedSignal

def get_timeWinsIntersect(itwA, itwB, lastPoint, firstPoint=0, ifDisplay=False, lfp=None):
    """
    return itwTot
    
    lfp to add if you want to check overlapping windows
    """
    skull=np.ones(lastPoint, bool)
    
    if np.logical_or(itwA.shape[0]==0, itwA.shape[1]==0):
        itwTot=itwB
    elif np.logical_or(itwB.shape[0]==0, itwB.shape[1]==0):
        itwTot=itwA
    else:
        for itw in itwA:
            skull[itw[0]:itw[1]]=False
        for itw in itwB:
            skull[itw[0]:itw[1]]=False

        edgeInds=np.where(np.diff(np.where(skull)[0])>1)[0]
        trues=np.where(skull)[0]

        itwTot=[[trues[edgeInds][i]+1, trues[edgeInds+1][i]] for i in range(edgeInds.shape[0])]

        if skull[0]==False: # starts by False (not detected window)
            itwTot.insert(0, [firstPoint, trues[0]])

        if skull[-1]==False:
            itwTot.append([trues[-1], lastPoint])

        itwTot=np.array(itwTot)

        if ifDisplay:
            lfpTpItwA=get_timeWinsTemplatedSignal(lfp, itwA)
            lfpTpItwB=get_timeWinsTemplatedSignal(lfp, itwB)
            lfpTpItwTot=get_timeWinsTemplatedSignal(lfp, itwTot)

            if np.sum(np.isnan(lfpTpItwA)==False)+np.sum(np.isnan(lfpTpItwB)==False) == np.sum(np.isnan(lfpTpItwTot)==False):
                print('NON-overlapping windows')
            else:
                print('Overlapping windows')

        if itwTot[0][0]>itwTot[0][1]:
            print('WARNING: firstPoint is greater than the first time window. Consider editing firstPoint.')
    return itwTot


def get_timeWins4TimeVect(timeVect, dtype = int):
    """
    return timeWins
    """
    timeWins = np.zeros((timeVect.shape[0], 2), dtype = dtype)
    for ind in range(timeVect.shape[0]-1):
        timeWins[ind, :] = [timeVect[0:-1][ind], timeVect[1:][ind]]
    timeWins = timeWins[0:-1]
    return timeWins

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ DETECT ----------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
def detect_movement_artifacts(chTraces, chGoodInds, tetInds, verbose=True,
                              SR=20000, thresh_mvA=3, extend_mvA_ms=200, threshCorr=0.6 ):
    """
    Detection of movement artifacts using vectorized operations.
    
    Parameters:
    - chTraces (ndarray): EEG/LFP data array (channels x time).
    - chGoodInds (list): Indices of good channels.
    - tetInds (list): List of tetrode indices.
    - SR (int): Sampling rate (default=20000 Hz).
    - thresh_mvA (float): Z-score threshold for movement artifacts.
    - extend_mvA_ms (int): Extension of detected artifacts (ms).
    - threshCorr (float): Correlation threshold for refinement.
    
    Returns:
    - badWins_refined (ndarray): Array of refined bad time windows.
    """
    # Compute gradient for all channels in one step
    if verbose:
        print('Compute gradient for all channels')
    gradients = np.gradient(chTraces[chGoodInds], axis=1)
    zgradients = np.abs(stats.zscore(gradients, axis=1))

    # Compute Z-score for the LFP signals
    if verbose:
        print('Compute Z-score for the LFP signals')
    zlfp = np.abs(stats.zscore(chTraces[chGoodInds], axis=1))

    # Identify bad points where both gradient and LFP exceed threshold
    badPoints = np.where((zgradients > thresh_mvA) & (zlfp > thresh_mvA))

    # Extend bad points using a smoothing kernel
    if verbose:
        print('Extend bad points using a smoothing kernel')
    extend_mvA_samples = int(extend_mvA_ms * SR / 1000)
    skull = np.zeros(chTraces.shape[1], dtype=bool)
    skull[badPoints[1]] = True
    kernel = np.ones(extend_mvA_samples)
    skull_smooth = np.convolve(skull, kernel, mode='same') > 0
    badPoints = np.where(skull_smooth)[0]

    if badPoints.size > 0:
        edges = np.where(np.diff(badPoints) > 1)[0] + 1  # Find breakpoints
        segments = np.split(badPoints, edges)  # Split into contiguous segments
        badWins = np.column_stack([(seg[0], seg[-1]) for seg in segments]).T  # Convert to (start, end)
    else:
        badWins = np.array([[1, 2]])  # Placeholder if no bad points

    # Add a condition on correlation between wires of a given tetrode
    tetWinCorrCoef=[]
    for teti, inds in enumerate(tetInds):
        winCorrCoef=np.array([  np.sum(np.triu( np.corrcoef(chTraces[inds, win[0]:win[1]]) , 1))/6  for win in badWins])
        tetWinCorrCoef.append(winCorrCoef)

    badWins_final=badWins[np.where(np.mean(tetWinCorrCoef, 0)<threshCorr)[0]]
    if verbose:
        print('Done movement artefacts')
    return badWins_final


    
def detect_subtle_artifacts(chTraces, chGoodInds, tetInds, chRegs, SR=20000, verbose=True,
                            extend_final_com_ms=200, lowcut_detPeaks=300, highcut_detPeaks=7000, order_detPeaks=2, 
                            thresh_dt1=10, nContacts_dt1=3, nTetrodes_dt1=2, extend_final_ms_dt1=20, 
                            thresh_peakDetect=4.5, window_ms_dt2=5, thresh_dt2=10, nContacts_dt2=2, nTetrodes_dt2=2, extend_final_ms_dt2=20,
                            ):
    """
    Detects subtle artifacts using bandpass filtering and peak detection.
    
    Parameters:
    - chTraces (ndarray): EEG/LFP data array.
    - chGoodInds (list): List of good channel indices.
    - tetInds (list): List of tetrode indices.
    - chRegs (list): Array mapping channels to regions.
    - SR (int): Sampling rate (default=20000 Hz).
    - lowcut_detPeaks (int): Low cut-off frequency for peak detection.
    - highcut_detPeaks (int): High cut-off frequency for peak detection.
    - order_detPeaks (int): Filter order.
    - thresh_dt1 (float): Z-score threshold for the first detector.
    - nContacts_dt1 (int): Minimum number of contacts required for detection in first detector.
    - nTetrodes_dt1 (int): Minimum number of tetrodes required for first detector.
    - extend_final_ms_dt1 (int): Extension window for first detector (ms).
    - thresh_peakDetect (float): Peak detection threshold.
    - window_ms_dt2 (int): Window size for second detector (ms).
    - thresh_dt2 (float): Threshold for spike count in the second detector.
    - nContacts_dt2 (int): Minimum number of contacts required for second detector.
    - nTetrodes_dt2 (int): Minimum number of tetrodes required for second detector.
    - extend_final_ms_dt2 (int): Extension window for second detector (ms).
    
    Returns:
    - bad_windows (ndarray): Array of detected bad time windows.
    
    Steps:
    - Filters signals in a high-frequency range (300-7000 Hz).
    - Identifies peaks crossing amplitude and gradient thresholds.
    - Uses channel and tetrode-level consensus to refine detection.
    """
    # First filter 300-7000 Hz to detect peaks
    if verbose:
        print('Apply bandpass filtering to isolate high-frequency components')
    chTraces_filt=bandpass_filter_bessel(chTraces, lowcut_detPeaks, highcut_detPeaks, SR, order=order_detPeaks)

    # Find correspondence between channel and tetrode
    ch2Tet=np.zeros(len(chRegs), int)
    for teti, chs in enumerate(tetInds):
        for ch in chs:
            ch2Tet[ch]=teti

    # Initial detection
    if verbose:
        print('Compute gradient and filtered signal Z-score for each channel')
    all_bad_points_per_channel_dt1=[]
    all_bad_points_per_channel_dt2=[]
    for chi, ch in enumerate(chGoodInds):
        if verbose:
            print(f" {chi + 1} / {len(chGoodInds)}")

        lfp=chTraces[ch]
        zgradient = np.abs(stats.zscore(np.gradient(lfp)))
        zlfp_filt=stats.zscore(chTraces_filt[ch])

        ## ------------------------------------ DETECTOR 1 ------------------------------------
        # Identify points exceeding gradient and amplitude (filtered signal) thresholds
        bad_indices = np.where((zgradient > thresh_dt1) & (np.abs(zlfp_filt) > thresh_dt1))[0]
        skull_tmp = np.zeros_like(lfp, dtype=bool)
        skull_tmp[bad_indices] = True
        all_bad_points_per_channel_dt1.append(np.where(skull_tmp)[0])

        ## ------------------------------------ DETECTOR 2 ------------------------------------
        # Make activity vector from peak detection
        peaks=detectPeaks(np.abs(zlfp_filt))
        thresh_cross=np.where(np.abs(zlfp_filt)>thresh_peakDetect)[0]
        peaksAbove=np.intersect1d(peaks, thresh_cross)
        actVector=np.zeros_like(zlfp_filt, int)
        actVector[peaksAbove]=1
        # Get spike by spike peri stimulus counts
        pkPSC=get_peri_stimulus_counts(actVector, window_ms=window_ms_dt2, sampling_rate=SR)
        pkTimes=np.where(actVector)[0]
        all_bad_points_per_channel_dt2.append(pkTimes[pkPSC>thresh_dt2])

    ## ------------------------------------ CONSENSUS ------------------------------------
    if verbose:
        print('Find consensus points')
    detAll_bad_points=[]
    for all_bad_points_per_channel, nContacts, nTetrodes, extend_final_ms in zip(
                        [all_bad_points_per_channel_dt1, all_bad_points_per_channel_dt2],
                        [nContacts_dt1, nContacts_dt2],
                        [nTetrodes_dt1, nTetrodes_dt2],
                        [extend_final_ms_dt1, extend_final_ms_dt2]):
        # Filter artefacts detected on at least 3 contacts
        all_bad_points_binary = np.zeros((len(chGoodInds), len(lfp)), dtype=bool)
        for i, bad_points in enumerate(all_bad_points_per_channel):
            all_bad_points_binary[i, bad_points] = True

        # Sum across channels to count how many contacts detect artefacts at each time point
        bad_points_sum = np.sum(all_bad_points_binary, axis=0)
        consensus_bad_points = np.where(bad_points_sum >= nContacts)[0]

        # Make sure detected events are on at least X different tetrodes
        consensus_bad_points_nbTets=np.array([ len(np.unique(ch2Tet[ chGoodInds[np.where(all_bad_points_binary[:, pt])[0]]  ])) 
                                              for pt in consensus_bad_points])
        consensus_bad_points_final=consensus_bad_points[consensus_bad_points_nbTets>=nTetrodes]

        all_bad_points_ = np.sort(consensus_bad_points_final)

        # Extend consensus bad points - First by a fixed time
        skull_tmp2 = np.zeros_like(lfp, dtype=bool)
        skull_tmp2[all_bad_points_] = True
        extend_final_samples = int(extend_final_ms * SR / 1000)
        skull_tmp2 = smooth_binary_array(skull_tmp2, extend_final_samples)
        all_bad_points_final_loop=np.where(skull_tmp2)[0]

        detAll_bad_points.append(all_bad_points_final_loop)

    # Identify artefact windows
    all_bad_points=np.sort( np.unique(np.concatenate(detAll_bad_points)))
    if all_bad_points.size > 0:
        bad_windows = get_timeWins4AdjacentPoints(all_bad_points)
    else:
        bad_windows = np.array([[1, 2]])  # Placeholder if no bad points are found

    # Final merge when windows overlap
    extend_final_com_samples= int( (extend_final_com_ms/1000)*SR )
    badWins_final, _=get_timeWins_mergeIfNext(bad_windows, extend_final_com_samples )

    # Output
    bad_duration = np.sum(badWins_final[:, 1] - badWins_final[:, 0]) / SR
    if verbose:
        print(f"\n        Total detected (subtle artefacts): {bad_duration:.2f} s")
        print('           equals :', np.round((bad_duration / (chTraces.shape[1]/SR))*100, 3), '% of the recording')    

    return badWins_final

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ PLOT ------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
def get_tickLocsNLabels_centered(tickMin, tickMax, nTicks, dtype = float, conversion = 1):
    """
    return tickLocs, tickLabels 
    
    convert will be multiplied to tickLabels 
    to convert sample_rate to secs: 1/sr
    to convert sample_rate to ms: 1000/sr
    & vice versa
    
    ex:
    get_tickLocsNLabels_centered(0, 4*1250, 5, convert = 1000/sr)
    return
    (array([   0., 1250., 2500., 3750., 5000.]),
     array([-2000., -1000.,    -0.,  1000.,  2000.]))
    """
    tickLocs = np.array(np.linspace(tickMin, tickMax, nTicks))
    m = np.mean([tickMin, tickMax])
    tickLabels = np.array(np.array([-(tickMax-m)+(x/(nTicks-1))*2*(tickMax-m) for x in range(nTicks)])*conversion, dtype = dtype)
    
    return tickLocs, tickLabels


def plot_chTraces(chTraces, sr, chInds = None, t_sample = None, chCols = None, win_s = 1, hspace = -200, title = False, legend = True, chLegend = None, xlabel = True, fontsize_legend = 20, lw=1, nTicks=5, roundTickLabels=2, locLegend='upper right', titleLegend=None, title_fontsizeLegend=20, lwLegend=1, alpha=1, xTickFS=20,labFS=30, yTickFS=12,bbox_to_anchor=None, ignoreCh=[]):
    """
    chInds is a np.array
    Adviced size: plt.figure(figsize=(30,nCh*1.5))
    
    legendLabels 
    bbox_to_anchor=(1.1, 1.05) is good
    """
    try:
        chTraces.shape[1]
        multipleCh = 1 
    except:
        multipleCh = 0
    
    if chInds is None:
        if multipleCh:
            inds = np.arange(chTraces.shape[0])
        else:
            inds = np.array([0])
    else:
        inds = chInds
        
    nCh = inds.shape[0]
    
    if chCols is None:
        colors = sb.color_palette('tab10', nCh)
    else:
        colors = chCols
        
    if t_sample is None:
        if multipleCh:
            t = np.random.choice(chTraces.shape[1])
        else:
            t = np.random.choice(chTraces.shape[0])
    else:
        t = t_sample
        
    if chLegend is None:
        chLegend = np.copy(inds)
    
    start = int(t-win_s*sr/2)
    end = int(t+win_s*sr/2)
    
    xLoc, xLabels=get_tickLocsNLabels_centered(0, win_s*sr, nTicks, conversion=1/sr)
    xLabels=np.round(xLabels ,roundTickLabels)
    
    for chii, chi in enumerate(inds):
        if multipleCh:
            if chi not in ignoreCh:
                plt.plot(chTraces[chi, start:end]+hspace*chii, color = colors[chii], label = chLegend[chii], lw=lw, alpha=alpha)
        else:
            plt.plot(chTraces[start:end]+hspace*chii, color = colors[chii], label = chLegend[chii], lw=lw, alpha=alpha)
    
    plt.xlim(0, win_s*sr)
    plt.xticks(xLoc, xLabels, fontsize = xTickFS)
    plt.yticks(fontsize=yTickFS)
    
    if legend:
        if bbox_to_anchor != None:
            leg = plt.legend(fontsize=fontsize_legend, loc = locLegend, title=titleLegend, title_fontsize=title_fontsizeLegend, bbox_to_anchor=bbox_to_anchor)
        else:
            leg = plt.legend(fontsize=fontsize_legend, loc = locLegend, title=titleLegend, title_fontsize=title_fontsizeLegend)
        for line in leg.get_lines():
            line.set_linewidth(lwLegend)
    if title:
        plt.title('t = '+str(t)+' sample points <=> '+str(t*1000/sr)+' ms')
    if xlabel:
        plt.xlabel('Time (secs)', fontsize = labFS)
        

def plot_detected_events(chTraces, chRegs, SR, badWinArray, badWini, ignoreCh=[], win_s=2, lw=1, hspace=-7000, extend_final_com_ms=200):
    
    tetWins=get_timeWins4TimeVect(np.arange(0, len(chRegs)+4, 4))
    chCols_tet=np.row_stack([[col]*4 for col in sb.color_palette('Dark2', len(tetWins))])
    
    t=badWinArray[badWini][0]

    chLegend=[str(i)+' '+reg for i,reg in enumerate(chRegs)]
    plt.figure(figsize=(20, len(chRegs)))
    plot_chTraces(chTraces, SR, win_s=win_s, t_sample=t, hspace=hspace,
                     chCols=chCols_tet, ignoreCh=ignoreCh, chLegend=chLegend, lw=lw)

    plt.title('t_sample='+str(t)+'  '+str(np.round(t/SR/60,2))+' mns')

    # DETECTED
    plt.axvline(win_s*SR/2)
    plt.axvline( win_s*SR/2+ badWinArray[badWini][1]-badWinArray[badWini][0] )

    # # PUTATIVE EXTEND
    plt.axvline( win_s*SR/2 - int(SR*(extend_final_com_ms/1000))  , color='grey')
    plt.axvline( win_s*SR/2+ badWinArray[badWini][1]-badWinArray[badWini][0] + int(SR*(extend_final_com_ms/1000))  , color='grey')

    
    

