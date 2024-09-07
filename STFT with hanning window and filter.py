import numpy as np
import matplotlib.pyplot as plt
import pytube as pyt
import wave
import math
import os
import imageio_ffmpeg as ffmpeg
import subprocess
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, iirnotch


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

# Function to design a Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the band-pass filter to data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to apply a notch filter to suppress a specific harmonic frequency
def notch_filter(data, freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    notch_freq = freq / nyquist
    b, a = iirnotch(notch_freq, quality_factor)
    y = lfilter(b, a, data)
    return y

def detect_fundamental_frequency_segment(segment, sample_rate):
    n = len(segment)
    audio_fft = np.fft.fft(segment)
    audio_freqs = np.fft.fftfreq(n, 1/sample_rate)

    # Take the magnitude of the FFT and consider only the positive frequencies
    fft_magnitude = np.abs(audio_fft[:n//2])
    freqs_positive = audio_freqs[:n//2]

    # Find the peak frequency, which is the fundamental frequency
    fundamental_idx = np.argmax(fft_magnitude)
    fundamental_frequency = freqs_positive[fundamental_idx]

    return fundamental_frequency
def detect_fundamental_frequencies(audio_data, sample_rate, segment_length=2048, overlap=0.5):
    # Calculate segment step based on overlap
    step_size = int(segment_length * (1 - overlap))
    num_segments = (len(audio_data) - segment_length) // step_size + 1

    fundamental_frequencies = []

    # Loop through each segment of the audio
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        segment = audio_data[start_idx:end_idx]

    # Detect the fundamental frequency for this segment
        fundamental_freq = detect_fundamental_frequency_segment(segment, sample_rate)
        fundamental_frequencies.append(fundamental_freq)

    return fundamental_frequencies

def normalize_audio(audio_data):
    """Normalize the audio signal to ensure it is audible."""
    max_value = np.max(np.abs(audio_data))
    if max_value > 0:
        normalized_audio = audio_data / max_value
        return normalized_audio * 32767  # Scale to int16 range
    else:
        return audio_data
    
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
        
        # Detect multiple fundamental frequencies from the audio (with 50% overlap)
        fundamental_frequencies = detect_fundamental_frequencies(audio_data, sample_rate, segment_length=2048, overlap=0.5)

        # Apply band-pass and notch filters for each segment's fundamental frequency
        filtered_audio = np.copy(windowed_audio)
        nyquist = 0.5 * sample_rate
        
        for fundamental_frequency in fundamental_frequencies:
            # Define filter parameters
            bandwidth = 50  # Bandwidth around the fundamental frequency

            lowcut = fundamental_frequency - bandwidth
            highcut = fundamental_frequency + bandwidth


            # Ensure lowcut and highcut are within valid range
            if lowcut <= 0:
                lowcut = 1  # Set to minimum possible valid frequency (e.g., 1 Hz)
            if highcut >= nyquist:
                highcut = nyquist - 1  # Ensure highcut is below the Nyquist frequency

            # Apply the band-pass filter for this segment if valid
            if lowcut < highcut:
                filtered_audio = bandpass_filter(filtered_audio, lowcut, highcut, sample_rate)

            # Apply the band-pass filter for this segment
            filtered_audio = bandpass_filter(filtered_audio, lowcut, highcut, sample_rate)

            # Apply notch filters for harmonics (2nd, 3rd, 4th, etc.)
            harmonics_to_remove = [2, 3]  # Remove 2nd, 3rd, 4th, 5th harmonics
            for harmonic in harmonics_to_remove:
                harmonic_freq = fundamental_frequency * harmonic
            if harmonic_freq < nyquist:
                        filtered_audio = notch_filter(filtered_audio, harmonic_freq, sample_rate)
            
            # Normalize the filtered audio to ensure it's audible
            filtered_audio = normalize_audio(filtered_audio)
            
        # Save the filtered audio
        wav.write('filtered_audio_no_harmonics.wav', sample_rate, filtered_audio.astype(np.int16))

    audio_array, frame_rate = analyze_audio('filtered_audio_no_harmonics.wav')
     
 
    #audio_array, frame_rate = analyze_audio(output_filename + '_windowed.wav')
    #audio_array, frame_rate = analyze_audio('filtered_audio.wav')
    audio_array, frame_rate = analyze_audio('filtered_audio_no_harmonics.wav')

    if audio_array is not None and frame_rate is not None:

        segment_length = 1024  # Define segment length for analysis
        notes = display_notes(audio_array, frame_rate, segment_length)

        for i, segment_notes in enumerate(notes):
            print(f"Segment {i+1}: {', '.join(segment_notes)}")

    else:
        print("Failed to analyze audio.")

if __name__ == "__main__":
    main()
