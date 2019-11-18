import aubio
import numpy as np  # importing array, ma, zeros, hstack, and nan
import matplotlib.pyplot as plt

downsample = 1
samplerate = 44100 // downsample

win_s = 4096 // downsample  # fft size
hop_s = 512 // downsample  # hop size


def get_waveform_plot_from_filename(filename, srate=0, block_size=4096, ax=None, dsample=2 ** 4):
    if not ax:
        fig = plt.figure(figsize=(8, 20))
        ax = fig.add_subplot(111)
    hop_size = block_size

    allsamples_max = np.zeros(0, )
    dsample = dsample  # to plot n samples / hop_s

    a = aubio.source(filename, srate, hop_size)  # source file
    if srate == 0:
        srate = a.samplerate

    total_frames = 0
    while True:
        samples, read = a()
        # keep some data to plot it later
        new_maxes = (abs(samples.reshape(hop_size // dsample, dsample))).max(axis=0)
        allsamples_max = np.hstack([allsamples_max, new_maxes])
        total_frames += read
        if read < hop_size: break
    allsamples_max = (allsamples_max > 0) * allsamples_max
    allsamples_max_times = [(float(t) / dsample) * hop_size for t in range(len(allsamples_max))]

    ax.plot(allsamples_max_times, allsamples_max, '-b')
    ax.plot(allsamples_max_times, -allsamples_max, '-b')
    ax.axis(xmin=allsamples_max_times[0], xmax=allsamples_max_times[-1])

    set_xlabels_sample2time(ax, allsamples_max_times[-1], srate)
    return ax


def get_waveform_plot(allsamples_max_times, allsamples_max, ax=None):
    if not ax:
        fig = plt.figure(figsize=(8, 20))
        ax = fig.add_subplot(111)

    ax.plot(allsamples_max_times, allsamples_max, '-b')
    ax.plot(allsamples_max_times, -allsamples_max, '-b')
    ax.axis(xmin=allsamples_max_times[0], xmax=allsamples_max_times[-1])

    set_xlabels_sample2time(ax, allsamples_max_times[-1], samplerate)
    return ax


def set_xlabels_sample2time(ax, latest_sample, srate):
    ax.axis(xmin=0, xmax=latest_sample)
    if latest_sample / float(srate) > 60:
        ax.set_xlabel('time (mm:ss)')
        ax.set_xticklabels(
            ["%02d:%02d" % (t / float(srate) / 60, (t / float(srate)) % 60) for t in ax.get_xticks()[:-1]],
            rotation=50)
    else:
        ax.set_xlabel('time (ss.mm)')
        ax.set_xticklabels(
            ["%02d.%02d" % (t / float(srate), 100 * ((t / float(srate)) % 1)) for t in ax.get_xticks()[:-1]],
            rotation=50)


def plot_from_pitches(timeframe, *pitches):
    times = [i * timeframe for i in range(pitches[0].shape[0])]
    fig = plt.figure(figsize=(16, 5))
    ax = plt.gca()
    for i in pitches:
        plt.plot(times, i)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('pitches')
    plt.show()
    return ax

