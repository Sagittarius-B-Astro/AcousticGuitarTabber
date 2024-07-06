import librosa
import numpy as np
import matplotlib.pyplot as plt


# load wave signal
wave_path = r"C:\Users\shaoq1\Documents\GitHub\test2\downloaded_audio.wav"
waveform,sample_rate = librosa.load(wave_path, sr=None)

# adding zeros
frame_size, hop_size = 1024, 512
if len(waveform) % hop_size != 0:
    frame_num = int((len(waveform)-frame_size)/hop_size) + 1
    pad_num = frame_num*hop_size + frame_size -len(waveform)
    waveform = np.pad(waveform, pad_width=(0,pad_num), mode="wrap")
frame_num = int((len(waveform)-frame_size)/hop_size) + 1

# divide into segments
row = np.tile(np.arrange(0,frame_size),(frame_num,1))
column = np.tile(np.arange(0,frame_num*(frame_size-hop_size),(frame_size-hop_size)),(frame_size,1)).T
index = row + column
waveform_frame = waveform[index]

# add frame
waveform_frame = waveform_frame * np.hanning(frame_size)

# fft
n_fft = 1024
waveform_stft = np.fft.rfft(waveform_frame, n_fft)

waveform_pow = np.abs(waveform_stft)**2/n_fft
waveform_db = 20 * np.log10(waveform_pow)

# graph

plt.figure(figsize=(10,10))
plt.imshow(waveform_db)
y_ticks = np.arrange(0, int(n_fft/2),100)
plt.yticks(ticks=y_ticks, labels=y_ticks*sample_rate/n_fft)
plt.title("Waveform_STFT")
plt.show()
print("done!")


