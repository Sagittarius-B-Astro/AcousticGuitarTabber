import numpy as np
import matplotlib.pyplot as plt
import pytube as pyt
import wave
import math
import os
import imageio_ffmpeg as ffmpeg
import subprocess
import scipy.io.wavfile as wav

'''
traceback: age restriction error
solution: find '_main_.py' under pytube pkg, change "ANDROID_EMBED" to "ANDROID_REACTOR"


traceback: pytube.exceptions.RegexMatchError: get_throttling_function_name: could not find match for multiple
solution: find 'cipher.py' under pytube pkg, change 

    r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&.*?\|\|\s*([a-z]+)',
    r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',

to

    r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&.*?\|\|\s*([a-z]+)',
    r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',

'''


def download_audio(youtube_url, output_filename):
    try:
        yt = pyt.YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True).first()
        file_path = output_filename + ".mp4"
        stream.download(filename=file_path)
        print(f"Downloaded file to {file_path}")

        # Use imageio-ffmpeg to get the path to the ffmpeg executable
        ffmpeg_path = ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg_path, '-i', file_path, output_filename + ".wav"]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command)
        
        print(f"Converted file to {output_filename}.wav")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
    except Exception as e:
        print(f"Error downloading or converting audio: {e}")


def analyze_audio(file_path):
    try:
        with wave.open(file_path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(num_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return audio_array, frame_rate
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None, None

def analyze_segment(audio_seg, frame_rate):
    fft_result = np.fft.fft(audio_seg)
    frequencies = np.fft.fftfreq(len(audio_seg), d=1/frame_rate)
    
    positive_frequencies = frequencies[:len(frequencies)//2]
    fft_result_positive = fft_result[:len(frequencies)//2]
    
    set_ = list(zip(positive_frequencies, np.abs(fft_result_positive)))
    set_.sort(key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in set_[:10]]

def hertz_to_note_name(frequency):
    A4_frequency = 440.0
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    
    if frequency <= 0:
        return "Invalid frequency"
    
    half_steps = round(12 * math.log2(frequency / A4_frequency))
    note_index = half_steps % 12
    octave = (half_steps + 9) // 12 if half_steps > -10 else - (abs(half_steps - 2) // 12)
    
    return f"{note_names[note_index]}{octave+4}" 

def frequency_to_chord(frequencies):
    chord_notes = [hertz_to_note_name(f) for f in frequencies]
    # Simple mapping for demonstration purposes
    # In reality, chord identification is more complex and requires considering musical context
    if "C" in chord_notes and "E" in chord_notes and "G" in chord_notes:
        return "C Major"
    elif "D" in chord_notes and "F#" in chord_notes and "A" in chord_notes:
        return "D Major"
    elif "E" in chord_notes and "G#" in chord_notes and "B" in chord_notes:
        return "E Major"
    elif "F" in chord_notes and "A" in chord_notes and "C" in chord_notes:
        return "F Major"
    elif "G" in chord_notes and "B" in chord_notes and "D" in chord_notes:
        return "G Major"
    elif "A" in chord_notes and "C#" in chord_notes and "E" in chord_notes:
        return "A Major"
    elif "B" in chord_notes and "D#" in chord_notes and "F#" in chord_notes:
        return "B Major"
    else:
        return "Unknown Chord"

def display_notes(audio_array, frame_rate, segment_length=1024):
    num_segments = len(audio_array) // segment_length
    notes = []

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        audio_segment = audio_array[start:end]
        dominant_frequencies = analyze_segment(audio_segment, frame_rate)
        segment_notes = [hertz_to_note_name(f) for f in dominant_frequencies]
        notes.append(segment_notes)

    return notes


def main():
    youtube_url = input("Enter YouTube video link: ") # use this link: https://www.youtube.com/watch?v=PH-buoH6dZA
    output_filename = "downloaded_audio"

    download_audio(youtube_url, output_filename)
   # audio_array, frame_rate = analyze_audio(output_filename + ".wav")

    sample_rate, audio_data = wav.read(output_filename + ".wav")

    if audio_data.ndim == 2:
        audio_data = audio_data[:, 0]
        
        window = np.hanning(len(audio_data))
        windowed_audio = audio_data * window
        
        wav.write(output_filename + '_windowed.wav', sample_rate, windowed_audio.astype(np.int16))

    audio_array, frame_rate = analyze_audio(output_filename + '_windowed.wav')

    if audio_array is not None and frame_rate is not None:

        segment_length = 1024  # Define segment length for analysis
        notes = display_notes(audio_array, frame_rate, segment_length)

        for i, segment_notes in enumerate(notes):
            print(f"Segment {i+1}: {', '.join(segment_notes)}")

    else:
        print("Failed to analyze audio.")

if __name__ == "__main__":
    main()
