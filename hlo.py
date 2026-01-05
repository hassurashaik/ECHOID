import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\Shaik\OneDrive\Desktop\EchoID\EchoID\models\cnn_model_v8\model_v8.keras"
AUDIO_PATH = "7780-274562-0000_chunk_04_Noise Gate (1).mp3"

SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TOP_K_CHANNELS = 6

CONV_LAYERS = ["conv2d", "conv2d_1", "conv2d_2"]

# =====================================================
# Load model
# =====================================================
model = load_model(MODEL_PATH)
model.summary()

print("\n📊 Number of channels per Conv layer:\n")
for layer in model.layers:
    if "conv" in layer.name.lower():
        print(f"{layer.name} → {layer.filters} channels")

# =====================================================
# Audio → Mel (ONE WINDOW)
# =====================================================
def audio_to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel -= mel.min()
    mel /= (mel.max() + 1e-9)
    return mel

audio, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)
mel = audio_to_mel(audio)
mel_input = mel[np.newaxis, ..., np.newaxis]

# =====================================================
# Prediction
# =====================================================
pred = model.predict(mel_input, verbose=0)[0][0]
label = "Target Speaker" if pred > 0.5 else "Other Speaker"
conf = pred if pred > 0.5 else 1 - pred

print(f"\nPrediction: {label} | Confidence: {conf:.3f}")

# =====================================================
# CHANNEL-WISE GRAD-CAM (LOGITS-BASED)
# =====================================================
def channelwise_gradcam(model, mel_input, layer_name, top_k):

    logits = model.layers[-1].input  # pre-sigmoid logits

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, logits]
    )

    with tf.GradientTape() as tape:
        conv_out, logit_out = grad_model(mel_input)
        loss = logit_out[:, 0]

    grads = tape.gradient(loss, conv_out)[0]   # (H, W, C)
    conv_out = conv_out[0]

    channel_scores = tf.reduce_mean(tf.abs(grads), axis=(0, 1))
    top_channels = tf.argsort(channel_scores, direction="DESCENDING")[:top_k]

    cams = []
    for ch in top_channels:
        cam = conv_out[:, :, ch] * grads[:, :, ch]
        cam = tf.maximum(cam, 0)
        cam -= tf.reduce_min(cam)
        cam /= (tf.reduce_max(cam) + 1e-9)
        cam = tf.pow(cam, 0.3)   # <-- KEY LINE

        cams.append(cam.numpy())

    return cams, top_channels.numpy(), channel_scores.numpy()

# =====================================================
# FEATURE MAPS (ACTIVATIONS)
# =====================================================
def get_feature_maps(model, mel_input, layer_name):
    feature_model = tf.keras.models.Model(
        model.inputs,
        model.get_layer(layer_name).output
    )
    return feature_model(mel_input)[0].numpy()

# =====================================================
# VISUALIZATION (ONE WINDOW)
# =====================================================
for layer in CONV_LAYERS:

    cams, channels, scores = channelwise_gradcam(
        model, mel_input, layer, TOP_K_CHANNELS
    )

    feature_maps = get_feature_maps(model, mel_input, layer)

    plt.figure(figsize=(18, 8))

    # Mel spectrogram
    plt.subplot(3, TOP_K_CHANNELS + 1, 1)
    plt.imshow(mel, cmap="magma")
    plt.title("Mel Spectrogram")
    plt.axis("off")

    # Channel activations
    for i, ch in enumerate(channels):
        plt.subplot(3, TOP_K_CHANNELS + 1, i + 2)
        plt.imshow(feature_maps[:, :, ch], cmap="magma")
        plt.title(f"{layer} | Ch {ch}\nActivation")
        plt.axis("off")

    # Channel Grad-CAMs
    for i, cam in enumerate(cams):
        plt.subplot(3, TOP_K_CHANNELS + 1, TOP_K_CHANNELS + i + 3)
        plt.imshow(cam, cmap="magma")
        plt.title(f"{layer} | Ch {channels[i]}\nGrad-CAM")
        plt.axis("off")

    plt.suptitle(
        f"Single Window – Channel Activations vs Grad-CAM ({layer})",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()

print("\n✅ SINGLE-WINDOW PER-CHANNEL ANALYSIS COMPLETED")
