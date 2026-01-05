# Voice Speaker Recognition - Audio Splitting Module
"""
Splits long audio recordings into fixed-duration WAV chunks
for speaker recognition model training.

Enhancement:
- Short final chunks are ZERO-PADDED

Project: EchoID
Purpose: Audio Chunk Creation (FLAC → WAV)
"""

import logging
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


# ------------------ Logger ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AudioChunker:
    """
    Splits audio files into fixed-duration WAV chunks.
    """

    def __init__(self, chunk_duration_sec=3, sample_rate=16000):
        self.chunk_duration_sec = chunk_duration_sec
        self.sample_rate = sample_rate

        logger.info(
            "Initialized AudioChunker | Chunk: %ds | SR: %dHz",
            chunk_duration_sec,
            sample_rate
        )

    # ---------------------------------------------------------
    def _create_audio_chunks(self, input_file, output_dir):
        """
        Create WAV chunks from a single FLAC file.
        """

        logger.info("📁 Loading audio: %s", input_file)

        audio, sr = librosa.load(
            input_file, sr=self.sample_rate, mono=True
        )

        samples_per_chunk = int(self.chunk_duration_sec * sr)
        total_samples = len(audio)
        total_chunks = int(np.ceil(total_samples / samples_per_chunk))

        logger.info(
            "Duration: %.2fs | Chunks: %d",
            total_samples / sr,
            total_chunks
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_file).stem

        for i in range(total_chunks):
            start = i * samples_per_chunk
            end = min(start + samples_per_chunk, total_samples)

            chunk = audio[start:end]

            if len(chunk) < samples_per_chunk:
                chunk = np.pad(
                    chunk,
                    (0, samples_per_chunk - len(chunk)),
                    mode="constant"
                )

            chunk_path = output_dir / f"{stem}_chunk_{i:02d}.wav"

            sf.write(
                chunk_path,
                chunk,
                sr,
                subtype="PCM_16"
            )

    # ---------------------------------------------------------
    def process_speaker(self, speaker_dir, output_dir):
        """
        Process all FLAC files for one speaker.
        """

        speaker_dir = Path(speaker_dir)
        output_dir = Path(output_dir)

        logger.info("🎤 Processing speaker: %s", speaker_dir.name)
        logger.info("=" * 50)

        flac_files = list(speaker_dir.glob("*.flac"))

        if not flac_files:
            logger.warning("⚠️ No FLAC files found in %s", speaker_dir)
            return

        for flac in flac_files:
            self._create_audio_chunks(flac, output_dir)

        logger.info("✅ Finished speaker: %s", speaker_dir.name)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    chunker = AudioChunker(
        chunk_duration_sec=3,
        sample_rate=16000
    )

    base_path = Path(
        "C:/Users/Shaik/OneDrive/Desktop/EchoID/EchoID/data"
    )

    for speaker in ["speaker0", "speaker1"]:
        chunker.process_speaker(
            speaker_dir=base_path / speaker,
            output_dir=base_path / speaker / "chunks"
        )
