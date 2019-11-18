import numpy as np


def dilation_by_volume(pitches, volume, threshold=0.65):
    new_pitches = np.copy(pitches)
    for i in range(new_pitches.shape[0] - 1):
        if volume[i] > threshold:
            new_pitches[i] = new_pitches[max(0, i + 1)]

    return new_pitches


def erosion(pitches: np.ndarray, frame_s=3, operation=np.mean):
    if frame_s < 2:
        return pitches
    new_pitches = list()
    for i in range(pitches.shape[0] - frame_s + 1):
        frame = pitches[i: i + frame_s]
        if True in np.isnan(frame):
            new_pitches += [np.nan]
        else:
            new_pitches += [operation(frame)]
    return np.pad(new_pitches, (0, pitches.shape[0] - len(new_pitches)))


def dilation(pitches: np.ndarray, frame_s=3, operation=np.nanmean):
    if frame_s < 2:
        return pitches
    new_pitches = list()
    for i in range(pitches.shape[0] - frame_s + 1):
        frame = pitches[i: i + frame_s]
        print(frame)
        if False not in np.isnan(frame):
            new_pitches += [np.nan]
        else:
            new_pitches += [operation(frame)]
    return np.pad(new_pitches, (0, pitches.shape[0] - len(new_pitches)))

