import sys
sys.path.append(".venv")

from scipy.io import wavfile
import numpy as np
import utils
import essentia.standard as es
import copy

from pitchDetection import pitchDetection
from onsetDetection import onsetDetection
from essentiaSegmentation import essentiaSegmentation

fileName="Humpty.wav"

pitchDetectionResolution=0.01 #In Must be at least as large as onsetDetectionResolution

onsetDetectionMinDistance=int(0.1/pitchDetectionResolution) #Having onsets be at least 0.1 seconds apart seems to be (inexplicably) the sweet spot, so this is hardcoded (for now at least)

sampleRate, audio = wavfile.read(fileName)
audio=audio.astype(np.float32)
essentiaAudio=es.EqloudLoader(filename=fileName, sampleRate=sampleRate)()

frequenciesList=pitchDetection(fileName, audio, essentiaAudio, sampleRate, pitchDetectionResolution)


onsetsList=[onsetDetection(essentiaAudio, sampleRate, pitchDetectionResolution, onsetDetectionMinDistance, frequencies) for frequencies in frequenciesList] #One will be none. Calculate onsets from all frequency generations

for onsets in onsetsList:

    notes = []

    for i, onset in enumerate(onsets): #Overload __next__
        frameStart = round(onset/pitchDetectionResolution)

        if i==len(onsets)-1:
            frameEnd=len(onset.frequencies)-1
        else:
            frameEnd = round(onsets[i+1]/pitchDetectionResolution)

        frame = onset.frequencies[frameStart:frameEnd]

        frameLength=len(frame)

        frame = list(filter(lambda _: not (np.isnan(_) or _>=0), frame))

        frameMidi=utils.hz_to_midi(np.median(frame))

        notes.append([frameStart*pitchDetectionResolution, frameMidi, frameLength*pitchDetectionResolution])

    notesCopy = copy.deepcopy(notes)

    notes = [[0, np.nan, 0]]

    for i, note in enumerate(notesCopy):  #Combining adjacent notes with the same MIDI number --- this removes the "jitter" or "clicking" that occurs when a single tone is split across multiple notes
        currentNote = notes[-1]

        if currentNote[1] == note[1]:
            currentNote[2]+=note[2]

        else:
            notes.append(note)

    #Writing the notes to file
    from midiutil.MidiFile import MIDIFile

    mf=MIDIFile(1,deinterleave=False, removeDuplicates=False)
    track=0
    channel=0
    volume=127
    bpm=120

    time=0
    mf.addTrackName(track, time, "Test")
    mf.addTempo(track, time, bpm)


    for note in notes:
        if (np.isnan(note[1])) or (note[2]==0):
            continue

        mf.addNote(track, channel, note[1], utils.convert_seconds_to_quarter(note[0],bpm), utils.convert_seconds_to_quarter(note[2],bpm), volume)


    with open(f"{onset.name[0]}_{onset.name[1]}.mid","wb") as outf:
        mf.writeFile(outf)

for frequencies in frequenciesList:
    essentiaSegmentation(essentiaAudio, frequencies)
