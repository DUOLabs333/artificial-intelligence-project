import numpy as np
def onsetDetection(onsetDetectionMethod, audio, sampleRate, pitchDetectionResolution, onsetDetectionMinDistance, frequencies):

    if onsetDetectionMethod=="none":
        onsets=[i*pitchDetectionResolution for i in range(len(frequencies))] #Baseline
    
    elif onsetDetectionMethod=="librosa":
        import librosa
        onsets=librosa.onset.onset_detect(y=audio, sr=sampleRate, units="time", backtrack=False)

    #CANOE
    elif onsetDetectionMethod=="roerich": #roerich. Don't use this unless you have a beefy computer ---- it will use all of your cores, with no way to change it
        import roerich.change_point
        
        config={"class": "ChangePointDetectionClassifier", "args": {"base_classifier": "mlp", "metric":"klsym", "step": 1, "n_runs":10}}

        onsets=getattr(roerich.change_point, config["class"])(**config["args"]).predict(frequencies)*pitchDetectionResolution

    elif onsetDetectionMethod=="ruptures": #ruptures
        import ruptures
        config={"class": "Binseg", "args":{"min_size": onsetDetectionMinDistance}}

        onsets=np.multiply(getattr(ruptures, config["class"])(**config["args"]).fit_predict(frequencies, pen=0),pitchDetectionResolution)

    return onsets

