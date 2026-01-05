import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import soundfile as sf
from keras.models import load_model
from scipy.ndimage import gaussian_filter1d

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\Shaik\OneDrive\Desktop\EchoID\EchoID\models\cnn_model_v9\model_v9.keras"
AUDIO_PATH = r"C:\Users\Shaik\OneDrive\Desktop\EchoID\EchoID\data\speaker1\7780-274562-0005.flac"

SR = 16000
WINDOW_DURATION = 3.08
WINDOW_SAMPLES = int(SR * WINDOW_DURATION)
STRIDE_SAMPLES = WINDOW_SAMPLES // 2

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
EXPECTED_FRAMES = 188
IG_STEPS = 50

IMPORTANCE_THRESHOLD = 0.4
SAVE_DIR = "highlighted_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)
model.summary()
print("✅ Model loaded")

# =====================================================
# MEL FUNCTION
# =====================================================
def audio_to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False
    )

    # 🔥 FORCE FIXED TIME DIMENSION (CRITICAL)
    if mel.shape[1] > EXPECTED_FRAMES:
        mel = mel[:, :EXPECTED_FRAMES]
    elif mel.shape[1] < EXPECTED_FRAMES:
        mel = np.pad(
            mel,
            ((0, 0), (0, EXPECTED_FRAMES - mel.shape[1])),
            mode="constant"
        )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    return mel_norm


# =====================================================
# INTEGRATED GRADIENTS
# =====================================================
def integrated_gradients(model, x, baseline, steps):
    grads_list = []
    for alpha in tf.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = model(interpolated, training=False)
            loss = pred[:, 0]
        grads = tape.gradient(loss, interpolated)
        grads_list.append(grads)

    avg_grads = tf.reduce_mean(tf.stack(grads_list), axis=0)
    ig = (x - baseline) * avg_grads
    ig = tf.reduce_mean(tf.abs(ig), axis=-1)[0]
    ig = ig / (tf.reduce_max(ig) + 1e-9)
    return ig.numpy()

# =====================================================
# LOAD FULL AUDIO
# =====================================================
audio, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)
audio_len = len(audio)

global_importance = np.zeros(audio_len)
coverage = np.zeros(audio_len)

baseline_input = None

# =====================================================
# SLIDING WINDOW IG
# =====================================================
for start in range(0, audio_len - WINDOW_SAMPLES, STRIDE_SAMPLES):

    chunk = audio[start:start + WINDOW_SAMPLES]
    mel = audio_to_mel(chunk)
    mel_input = mel[np.newaxis, ..., np.newaxis]

    if baseline_input is None:
        baseline_input = tf.zeros_like(mel_input)

    ig = integrated_gradients(model, mel_input, baseline_input, IG_STEPS)

    time_importance = gaussian_filter1d(ig.mean(axis=0), sigma=2)
    time_importance /= (time_importance.max() + 1e-9)

    sample_importance = np.repeat(time_importance, HOP_LENGTH)
    sample_importance = sample_importance[:WINDOW_SAMPLES]

    global_importance[start:start + len(sample_importance)] += sample_importance
    coverage[start:start + len(sample_importance)] += 1

# =====================================================
# NORMALIZE GLOBAL IMPORTANCE
# =====================================================
coverage[coverage == 0] = 1
global_importance /= coverage
global_importance /= (global_importance.max() + 1e-9)

# =====================================================
# SOFT MASK
# =====================================================
soft_mask = np.clip(
    (global_importance - IMPORTANCE_THRESHOLD) / (1 - IMPORTANCE_THRESHOLD),
    0,
    1
)

highlighted_audio = audio * soft_mask

# =====================================================
# SAVE AUDIO
# =====================================================
sf.write(os.path.join(SAVE_DIR, "highlighted_speech_full.wav"),
         highlighted_audio, SR)

print("✅ Highlighted full audio saved")

# =====================================================
# MEL SPECTROGRAMS (FULL AUDIO)
# =====================================================
mel_full = audio_to_mel(audio)
mel_highlighted = audio_to_mel(highlighted_audio)

# =====================================================
# VISUALIZATION (2×2)
# =====================================================
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(mel_full, aspect="auto", origin="lower", cmap="magma")
plt.title("Original Mel Spectrogram")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mel_highlighted, aspect="auto", origin="lower", cmap="magma")
plt.title("IG-Highlighted Mel Spectrogram")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(audio)
plt.title("Original Audio Waveform")

plt.subplot(2, 2, 4)
plt.plot(highlighted_audio)
plt.title("Sliding-Window IG Highlighted Audio")

plt.suptitle("Sliding-Window Integrated Gradients – Full Audio Explanation",
             fontsize=14)

plt.tight_layout()
plt.show()
