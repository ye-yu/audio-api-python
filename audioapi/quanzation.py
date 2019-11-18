import pandas as pd
import numpy as np


def quantize(pitches, threshold=0.6):
    audio = pd.DataFrame(data=dict(
        pitches=pitches
    ))
    audio['pitches ffill'] = audio['pitches'].fillna(method='ffill').fillna(0)
    audio['pitch diff'] = audio['pitches ffill'].diff().abs()
    audio['is pitch'] = ~np.isnan(audio['pitches'])

    # identifying groups
    groups = list()
    group_curr = 0
    prev_row = None
    for i, r in audio.iterrows():
        diff = r['pitch diff']
        is_pitch = r['is pitch']
        if diff > threshold or (prev_row is not None and not prev_row['is pitch'] and is_pitch):
            group_curr += 1
        if not is_pitch:
            groups += [0]
        else:
            groups += [group_curr]
        prev_row = r

    # group labelling
    audio['pitch group'] = groups
    audio['pitch group'] = audio['pitch group'].astype(float)
    audio[audio['pitch group'] < 0.5] = np.nan

    # averaging and mapping to quantized pitch
    quantized_df = audio.groupby('pitch group').mean()['pitches'].round(0).astype(int)
    quantized = list()
    for i, r in audio.iterrows():
        try:
            grp = int(r['pitch group'])
            quantized += [quantized_df[grp]]
        except ValueError as e:
            quantized += [np.nan]
    audio['quantized pitch'] = quantized

    return audio['quantized pitch'].to_numpy()
