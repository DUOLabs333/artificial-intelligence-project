import essentia.standard as es

def essentiaSegmentation(essentiaAudio, frequencies): #Use the PitchSegmentation code to compare my custom segmentation algorithm against

        onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(frequencies[1], essentiaAudio)

        import mido

        PPQ = 96 # Pulses per quarter note.
        BPM = 120 # Assuming a default tempo in Ableton to build a MIDI clip.
        tempo = mido.bpm2tempo(BPM) # Microseconds per beat.

        # Compute onsets and offsets for all MIDI notes in ticks.
        # Relative tick positions start from time 0.
        offsets = onsets + durations
        silence_durations = list(onsets[1:] - offsets[:-1]) + [0]

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        for note, onset, duration, silence_duration in zip(list(notes), list(onsets), list(durations), silence_durations):
            track.append(mido.Message('note_on', note=int(note), velocity=64,
                                      time=int(mido.second2tick(duration, PPQ, tempo))))
            track.append(mido.Message('note_off', note=int(note),
                                      time=int(mido.second2tick(silence_duration, PPQ, tempo))))

        mid.save(f"{frequencies[0]}_essentia.mid")
