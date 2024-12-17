#NOTE: Maybe add confidence threshold (whether confidence in pitch prediction or voiced prediction)? Nah. Instead, just use mode of quantized midi note and see if a high enough proportion is the same note

import librosa
import numpy
def pitchDetection(pitchDetectionMethod, audio, sampleRate, pitchDetectionResolution):

    if pitchDetectionMethod == "yin":
        import librosa

        frequencies=librosa.yin(audio, sr=sampleRate, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C9'), hop_length=round(pitchDetectionResolution*sampleRate))
    elif pitchDetectionMethod in ["yin", "pyin", "salience", "swipe", "swipe_slim"]:
        setattr(numpy, "NaN", numpy.nan)
        setattr(numpy, "bool8", numpy.bool_)
        
        import libf0
        import librosa
        frequencies, _, _= getattr(libf0, pitchDetectionMethod)(audio, sampleRate, H=round(pitchDetectionResolution*sampleRate), F_min=librosa.note_to_hz('C2'), F_max=librosa.note_to_hz('C7'))
    elif pitchDetectionMethod=="crepe":
        import crepe
        frequencies, _, _, _= crepe.predict(audio, sampleRate, step_size=1)
    
    """
    NOTE: Potential future method --- find good audio similarity metric. Then, can compare each snippet of sound (segmented using an external onset function, or internally by just increasing the length of snippet used until the distribution suddenly changes) against clips of pure tones, and see which tone matches it the best

    Candidates:
    Audio-Similarity
    CDPAM
    zimtorhli
    Conch-sounds
    nomad
    """
    return frequencies



    
    

