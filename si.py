import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\Shaik\OneDrive\Desktop\EchoID\EchoID\models\cnn_model_v9\model_v9.keras"
AUDIO_PATH = r"C:\Users\Shaik\OneDrive\Desktop\EchoID\EchoID\data\speaker1\7780-274562-0005.flac"

SR = 16000
WINDOW_DURATION = 3.08
WINDOW_SAMPLES = int(SR * WINDOW_DURATION)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
EXPECTED_FRAMES = 188

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# =====================================================
# AUDIO → MEL (FIXED SHAPE)
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

    if mel.shape[1] > EXPECTED_FRAMES:
        mel = mel[:, :EXPECTED_FRAMES]
    elif mel.shape[1] < EXPECTED_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, EXPECTED_FRAMES - mel.shape[1])))

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    return mel_norm

# =====================================================
# SALIENCY MAP
# =====================================================
def compute_saliency_map(model, x):
    x = tf.convert_to_tensor(x)

    # Build model that outputs logits
    logit_model = tf.keras.Model(
        model.inputs,
        model.layers[-1].input
    )

    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = logit_model(x, training=False)
        loss = logits[:, 0]

    grads = tape.gradient(loss, x)
    saliency = tf.abs(grads)

    saliency = tf.reduce_mean(saliency, axis=-1)[0]
    saliency = saliency / (tf.reduce_max(saliency) + 1e-9)

    return saliency.numpy()


# =====================================================
# MAIN
# =====================================================
audio, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)

chunk = audio[:WINDOW_SAMPLES]
if len(chunk) < WINDOW_SAMPLES:
    chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

mel = audio_to_mel(chunk)
mel_input = mel[np.newaxis, ..., np.newaxis]

# Prediction
pred = model.predict(mel_input, verbose=0)[0][0]
label = "Target Speaker" if pred > 0.5 else "Other Speaker"
conf = pred if pred > 0.5 else 1 - pred

print(f"Prediction → {label} (conf={conf:.2f})")

# Saliency
saliency_map = compute_saliency_map(model, mel_input)

# =====================================================
# VISUALIZATION
# =====================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(mel, cmap="magma")
plt.title("Mel Spectrogram")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap="inferno")
plt.title("Saliency Map")
plt.axis("off")

plt.suptitle(
    f"Saliency Analysis | {label} (conf={conf:.2f})",
    fontsize=14
)
plt.tight_layout()
plt.show()
