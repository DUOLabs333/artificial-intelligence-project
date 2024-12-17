import librosa
import numpy as np
def convert_seconds_to_quarter(time_in_sec, bpm):
    quarter_per_second = (bpm/60)
    time_in_quarter = time_in_sec * quarter_per_second
    return time_in_quarter

def hz_to_midi(hz):
    if np.isnan(hz):
        return hz
    return int(round(librosa.hz_to_midi(hz)))


def samplesToSeconds(num_samples, sampleRate):
    return num_samples/sampleRate

def secondsToSamples(num_seconds, sampleRate):
    return round(num_seconds*sampleRate)
