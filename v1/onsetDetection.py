import essentia.standard as es
import essentia

import numpy as np
import roerich
import ruptures
import utils

class Onsets(object):
    def __init__(self, frequencies, name, onsets):
        self.name=(frequencies[0], name)
        self.frequencies=frequencies[1]
        self.onsets=onsets

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.onsets)

    
def onsetDetection(essentiaAudio, sampleRate, pitchDetectionResolution, onsetDetectionMinDistance, frequencies):
    
    onsetsList = []
    #Essentia
    detection_functions={method: es.OnsetDetection(method=method, sampleRate=sampleRate) for method in ["hfc", "complex", "complex_phase", "flux", "melflux", "rms"]}
    
    
    # We need the auxilary algorithms to compute magnitude and phase.
    w = es.Windowing(type='hann')
    fft = es.FFT() # Outputs a complex FFT vector.
    c2p = es.CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.
    
    # Compute both ODF frame by frame. Store results to a Pool.
    pool = essentia.Pool()

    frameSize = 1024
    for frame in es.FrameGenerator(essentiaAudio, frameSize=frameSize, hopSize=512):
        magnitude, phase = c2p(fft(w(frame)))
        for key, func in detection_functions:
            pool.add(key, func(magnitude, phase))
    
    onsets=es.Onsets(frameRate=sampleRate/frameSize)(essentia.array(detection_functions.values()), [1/len(detection_functions) for _ in detection_functions])

    onsetsList.append(Onsets(frequencies, "Onsets", onsets))

    #SuperFluxExtractor
    
    onsets=es.SuperFluxExtractor(sampleRate=sampleRate)(essentiaAudio)

    onsetsList.append(Onsets(frequencies, "SuperFluxExtractor", onsets))

    #CANOE

    #roerich
    config={"class": "ChangePointDetectionClassifier", "args": {"base_classifier": "mlp", "metric":"klsym", "step": 1, "n_runs":10}}

    onsets=utils.getattr(roerich.change_point, config["class"])(**config["args"]).predict(frequencies[1])*pitchDetectionResolution

    onsetsList.append(Onsets(frequencies, "CanoeRoerich", onsets))

    
    #ruptures

    config={"class": "Pelt", "args":{"min_size": onsetDetectionMinDistance}}

    onsets=getattr(ruptures, config["class"])(**config["args"]).fit_predict(frequencies[1])*pitchDetectionResolution

    onsetsList.append(Onsets(frequencies, "CanoeRuptures", onsets))

    return onsetsList

