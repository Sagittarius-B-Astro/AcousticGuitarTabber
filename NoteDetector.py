import numpy as np
import matplotlib.pyplot as plt
!pip install pytube3
import pytube as pyt
import scipy as sp

def __main__():
  print("Enter video link to tabulate")

def mp3towav():
  yt = pyt.YouTube('https://www.youtube.com/watch?v=PH-buoH6dZA').streams.filter(only_audio=True).first()
  yt.download()

import wave
with wave.open("/Music WAV Files/CTY.wav", "rb") as wav_file: # Open the WAV file

    num_channels = wav_file.getnchannels() # Get the number of audio channels (1 for mono, 2 for stereo)
    print("Number of channels:", num_channels)

    sample_width = wav_file.getsampwidth() # Get the sample width (in bytes)
    print("Sample width (bytes):", sample_width)

    frame_rate = wav_file.getframerate() # Get the frame rate (number of samples per second)
    print("Frame rate (samples per second):", frame_rate)

    num_frames = wav_file.getnframes() # Get the total number of frames in the audio file
    print("Number of frames:", num_frames)

    audio_data = wav_file.readframes(num_frames) # Read all frames from the WAV file

# The audio data is read as bytes. You need to convert it to a format that you can work with, such as NumPy arrays.
# For example, if you want to convert the audio data to a NumPy array of 16-bit integers:
import numpy as np

audio_array = np.frombuffer(audio_data, dtype=np.int16) # Convert the audio data to a NumPy array of 16-bit integers

print("Shape of audio array:", audio_array.shape) # Now you can work with the audio data as a NumPy array

c=1
X = []
for i in range(2019_000, 2020_000):
  X.append(np.mean(audio_array[i:i+c]))

plt.plot(X)

def analyze_segment(audio_seg):
  # Assuming X_t is your array function

  fft_result = np.fft.fft(audio_seg) # Perform FFT on X_t

  # Frequency resolution (sampling frequency divided by number of samples)
  sampling_frequency = 48000  # Assuming 1 Hz sampling frequency for simplicity
  frequency_resolution = sampling_frequency / len(audio_seg)

  frequencies = np.fft.fftfreq(len(audio_seg), d=1/sampling_frequency)  # Generate frequencies corresponding to FFT result

  positive_frequencies = frequencies[:len(frequencies)//2] # Filter out negative frequencies (for real signals, only positive frequencies are meaningful)
  fft_result_positive = fft_result[:len(frequencies)//2]

  # Plot magnitude spectrum

  #plt.figure(figsize=(10, 5))
  plt.plot(positive_frequencies, np.abs(fft_result_positive))
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude')
  plt.title('Magnitude Spectrum')
  plt.grid(True)
  plt.show()

  set = list(zip(positive_frequencies, fft_result_positive))
  set.sort(key=lambda x : x[1])

  return [hertz_to_note(x[0]) for x in set[:10]]

audio_seg = audio_array[2019_000: 2020_000]
analyze_segment(audio_seg)

import math

def hertz_to_note(frequency):
    # Define the reference frequency for A4 (the A above middle C)
    A4_frequency = 440.0  # Hz

    # Define the names of the musical notes
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Calculate the number of half-steps away from A4
    half_steps = 12 * math.log2(frequency / A4_frequency)

    # Calculate the index of the closest note
    note_index = round(half_steps) % 12

    # Calculate the octave number
    octave = int((round(half_steps) + 9) // 12)

    # Return the note name and octave number
    return f"{note_names[note_index]}{octave}"

# Example usage:
frequency = 49.2  # A4
note = hertz_to_note(frequency)
print("Frequency:", frequency, "Hz")
print("Note:", note)