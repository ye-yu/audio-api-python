import aubio
import os
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


class Tone:
    def __init__(self, filename, tolerance=0.8, srate=None):
        if srate:
            self.s = aubio.source(filename, srate, hop_s)
        else:
            self.s = aubio.source(filename, samplerate, hop_s)
        self.samplerate = self.s.samplerate
        self.tolerance = tolerance
        self.pitch_o = aubio.pitch("yin", win_s, hop_s, self.samplerate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(tolerance)

        self.pitches = []
        self.confidences = []
        self.allsamples_max = np.zeros(0, )
        self.allsamples_max_times = None
        self.timeframe = 0
        self.times = None

        self.analysed = False

    def analyse(self):
        total_frames = 0
        while True:
            samples, read = self.s()
            if not self.timeframe:
                self.timeframe = read / self.samplerate

            # obtaining peak volumes
            new_maxes = (abs(samples.reshape(hop_s // downsample, downsample))).max(axis=0)
            self.allsamples_max = np.hstack([self.allsamples_max, new_maxes])
            self.allsamples_max_times = [(float(t) / downsample) * hop_s for t in range(len(self.allsamples_max))]

            # obtaining pitches and confidences
            pitch = self.pitch_o(samples)[0]
            confidence = self.pitch_o.get_confidence()
            if confidence < self.tolerance:
                pitch = np.nan
            self.pitches += [pitch]
            self.confidences += [confidence]
            total_frames += read
            if read < hop_s:
                break

        skip = 1

        self.pitches = np.array(self.pitches[skip:]).astype(float)
        self.confidences = np.array(self.confidences[skip:])
        self.times = [t * hop_s for t in range(len(self.pitches))]

        self.pitches[np.isfinite(self.pitches)] = np.where(
            self.pitches[np.isfinite(self.pitches)] < 30,
            np.nan,
            self.pitches[np.isfinite(self.pitches)]
        )
        self.pitches[np.isfinite(self.pitches)] = np.where(
            self.pitches[np.isfinite(self.pitches)] > 85,
            np.nan,
            self.pitches[np.isfinite(self.pitches)]
        )
        # similar to
        # self.pitches[self.pitches < 30] = np.nan
        # self.pitches[self.pitches > 85] = np.nan

        self.analysed = True

    def get_features(self, attr):
        if not self.analysed:
            return

        ret = dict()
        if attr == 'all':
            attr = ['pitches', 'volume', 'timeframe']
        if 'pitches' in attr:
            ret['pitches'] = self.pitches

        if 'volume' in attr:
            ret['volume'] = self.allsamples_max[1:]

        if 'timeframe' in attr:
            ret['timeframe'] = self.timeframe

        return ret

    def plot(self):
        filename = 'temp'
        fig = plt.figure(figsize=(15, 20))

        ax1 = fig.add_subplot(311)
        ax1 = get_waveform_plot(self.allsamples_max_times, self.allsamples_max, ax=ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)

        def array_from_text_file(fname, dtype='float'):
            fname = os.path.join(os.path.dirname(__file__), fname)
            return np.array([line.split() for line in open(fname).readlines()],
                            dtype=dtype)

        ax2 = fig.add_subplot(312, sharex=ax1)
        ground_truth = os.path.splitext(filename)[0] + '.f0.Corrected'
        if os.path.isfile(ground_truth):
            ground_truth = array_from_text_file(ground_truth)
            true_freqs = ground_truth[:, 2]
            true_freqs = np.ma.masked_where(true_freqs < 2, true_freqs)
            true_times = float(self.samplerate) * ground_truth[:, 0]
            ax2.plot(true_times, true_freqs, 'r')
            ax2.axis(ymin=0.9 * true_freqs.min(), ymax=1.1 * true_freqs.max())
        # plot raw pitches
        ax2.plot(self.times, self.pitches, '.g')
        # plot cleaned up pitches
        cleaned_pitches = self.pitches
        cleaned_pitches = np.ma.masked_where(self.confidences < self.tolerance, cleaned_pitches)
        ax2.plot(self.times, cleaned_pitches, '.-')
        plt.setp(ax2.get_xticklabels(), visible=True)
        ax2.set_ylabel('f0 (midi)')

        # plot confidence
        ax3 = fig.add_subplot(313, sharex=ax1)
        # plot the confidence
        ax3.plot(self.times, self.confidences)
        # draw a line at tolerance
        ax3.plot(self.times, [self.tolerance] * len(self.confidences))
        ax3.axis(xmin=self.times[0], xmax=self.times[-1])
        ax3.set_ylabel('confidence')
        set_xlabels_sample2time(ax3, self.times[-1], self.samplerate)
        plt.show()

    def reload(self, filename, tolerance, srate=None):
        self.__init__(filename, tolerance, srate)


if __name__ == "__main__":
    import pandas as pd
    audiofile = Tone('../test/sing-03.wav')
    audiofile.analyse()
    audiofile.plot()
    audiofile_feature = audiofile.get_features('all')
    print(pd.DataFrame(data=audiofile_feature))
