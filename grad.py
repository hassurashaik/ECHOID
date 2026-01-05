import os
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
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

THRESHOLD = 0.2
SEGMENT_DURATION = 0.30

SAVE_DIR = "highlighted_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model(MODEL_PATH)
model.summary()
print("✅ Model loaded")

# =====================================================
# AUDIO → MEL
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
# GRAD-CAM (FINAL CONV)
# =====================================================
FINAL_CONV_LAYER = "conv2d_2"

def compute_gradcam(model, mel_input):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(FINAL_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(mel_input)
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-9)

    return heatmap.numpy()

# =====================================================
# OVERLAY
# =====================================================
def overlay_gradcam(mel, heatmap):
    heatmap = cv2.resize(heatmap, (mel.shape[1], mel.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mel_img = np.uint8(255 * mel)
    mel_img = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(mel_img, 0.6, heatmap, 0.4, 0)

# =====================================================
# MAIN PIPELINE
# =====================================================
audio, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)
num_windows = int(np.ceil(len(audio) / WINDOW_SAMPLES))

start_sample = 0
window_index = 0

while start_sample < len(audio):

    chunk = audio[start_sample:start_sample + WINDOW_SAMPLES]
    if len(chunk) < WINDOW_SAMPLES:
        chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

    mel = audio_to_mel(chunk)
    mel_input = mel[np.newaxis, ..., np.newaxis]

    pred = model.predict(mel_input, verbose=0)[0][0]
    label = "Target Speaker" if pred > 0.5 else "Other Speaker"
    conf = pred if pred > 0.5 else 1 - pred

    print(f"🟦 Window {window_index} → {label} (conf={conf:.2f})")

    heatmap = compute_gradcam(model, mel_input)
    overlay = overlay_gradcam(mel, heatmap)

    # ================= AUDIO SAVING =================
    time_importance = np.mean(heatmap, axis=0)
    time_importance /= (time_importance.max() + 1e-9)

    important_frames = np.where(time_importance > THRESHOLD)[0]
    print(f"   → Highlighted frames: {len(important_frames)}")

    for frame in important_frames:
        t_local = frame * HOP_LENGTH / SR
        global_time = (start_sample / SR) + t_local

        start = int(global_time * SR)
        end = start + int(SEGMENT_DURATION * SR)

        start = max(0, start)
        end = min(len(audio), end)

        if end - start > int(0.05 * SR):
            sf.write(
                os.path.join(SAVE_DIR, f"window{window_index}_t{global_time:.2f}s.wav"),
                audio[start:end],
                SR
            )

    # ================= VISUALIZATION =================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(mel, cmap="magma")
    plt.title("Mel Spectrogram")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="magma")
    plt.title("Grad-CAM (Final Conv)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(f"Window {window_index} | {label} (conf={conf:.2f})")
    plt.tight_layout()
    plt.show()

    start_sample += WINDOW_SAMPLES
    window_index += 1

print("\n✅ Visualization + highlighted audio saving completed")
