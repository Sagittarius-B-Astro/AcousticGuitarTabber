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
row = np.tile(np.arange(0,frame_size),(frame_num,1))
column = np.tile(np.arange(0,frame_num*(frame_size-hop_size),(frame_size-hop_size)),(frame_size,1)).T
index = row + column
waveform_frame = waveform[index]

# add frame
waveform_frame = waveform_frame * np.hanning(frame_size)

# fft power to dB
n_fft = 1024
waveform_stft = np.fft.rfft(waveform_frame, n_fft)

waveform_pow = np.abs(waveform_stft)**2/n_fft
waveform_db = 20 * np.log10(waveform_pow)

# waveform_mag = np.abs(waveform_stft)


# max_freq = 2000
# max_index = int(max_freq / (sample_rate / n_fft))

# graph

plt.figure(figsize=(10,10))
# plt.imshow(waveform_mag[:, :max_index], aspect='auto', origin='lower', extent=[0, frame_num, 0, max_freq])

plt.imshow(waveform_db.T, aspect='auto', origin='lower')


y_ticks = np.arange(0, int(n_fft/2),50)
#y_ticks = np.arange(0, max_index, 5)
plt.yticks(ticks=y_ticks, labels=y_ticks*sample_rate/n_fft)


x_ticks = np.arange(0, frame_num, step=1000)  # Adjust step for more/less granularity
x_tick_labels = x_ticks * hop_size / sample_rate
plt.xticks(ticks=x_ticks, labels=np.round(x_tick_labels, 2))


# Labels and title
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Waveform_STFT")


plt.show()
#plt.savefig('plot.png', dpi = 300, bbox_inches='tight')
print("done!")


