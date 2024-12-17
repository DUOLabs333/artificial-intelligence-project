import numpy as np
import essentia.standard as es

#NOTE: Maybe add confidence threshold (whether confidence in pitch prediction or voiced prediction)? Nah. Instead, just use mode of quantized midi note and see if a high enough proportion is the same note

def pitchDetection(fileName, audio, essentiaAudio, sampleRate, pitchDetectionResolution):

    frequenciesList=[]

    frameSize=round(sampleRate*pitchDetectionResolution)
    hopSize=frameSize//4

    args = {"hopSize": hopSize, "frameSize": frameSize, "sampleRate": sampleRate}

    #MultiPitch*

    for algorithm in ["Kalpuri", "Melodia"]:
        pitches = getattr(es, "MultiPitch"+algorithm)(numberHarmonics=1, **args)(essentiaAudio)

        assert len(pitches[0])==1, f"{pitches[0]}"

        frequenciesList.append(["MultiPitch"+algorithm,np.flatten(pitches)])


    #PitchMelodia
    pitches, _ = es.PitchMelodia(numberHarmonics=1, **args)(essentiaAudio)
    assert len(pitches[0])==1, f"{pitches[0]}"

    frequenciesList.append(["PitchMelodia", np.flatten(pitches)])

    #pYIN and Melodia
    for algorithm in ["PredominantPitchMelodia", "PitchYinProbabilistic"]:
        pitches, _ = getattr(es, algorithm)(**args)(essentiaAudio)

        frequenciesList.append([algorithm, pitches])
    
    def gen_frame():
        return es.FrameGenerator(essentiaAudio, frameSize=frameSize, hopSize=hopSize, startFromZero=True, lastFrameToEndOfFile=True)

    #YIN
    pitches=[]
    for frame in gen_frame():
        pitch, confidence = es.PitchYin(frameSize=frameSize, sampleRate=sampleRate)(frame)

        if confidence==0:
            pitch=np.nan

        pitches.append(pitch)

    frequenciesList.append(["PitchYin", pitches])

    pitches=[]
    for frame in gen_frame():
        frame = es.Windowing(normalized=False, type="hann")
        frame=es.Spectrum(size=frameSize)(frame)
        pitch, confidence = es.PitchYinFFT(frameSize=frameSize, sampleRate=sampleRate)(essentiaAudio)

        if confidence == 0:
            pitch = np.nan
        pitches.append(pitch)

    frequenciesList.append(["PitchYinFFT", pitches])

    
    return frequenciesList


    



    
    

