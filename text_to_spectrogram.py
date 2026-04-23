#!/usr/bin/env python3
"""
Spectrogram Stego Encoder + Inspector (Text / Image / Inspect)
--------------------------------------------------------------
Modes:
1) Text  -> hide text in spectrogram -> Griffin-Lim -> audio -> optional embed into carrier
2) Image -> hide image in spectrogram -> Griffin-Lim -> audio -> optional embed into carrier
3) Inspect -> analyze an audio spectrogram to see if it likely contains an embedded message/image

Outputs for modes 1 & 2:
- final audio (.wav)
- final spectrogram image (.png)
- template image used for embedding (_template.png)

Outputs for mode 3:
- inspected spectrogram image (.png)
- threshold mask image (_mask.png)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write

# ---------- Constants ----------
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256

MIN_WIDTH = 400
MAX_WIDTH = 4000
FONT_SIZE = 180

EMBED_MIX_LEVEL = 4.0  


def render_text_image(text, min_width, max_width, height, font_size=48):
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy_img = Image.new("L", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy_img)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    raw_width = int(text_w * 1.2)
    width = max(min_width, min(raw_width, max_width))

    pad = 20
    tight_w = text_w + 2 * pad
    tight_h = text_h + 2 * pad
    tight_img = Image.new("L", (tight_w, tight_h), color=0)
    draw_tight = ImageDraw.Draw(tight_img)
    draw_tight.text((pad, pad), text, fill=255, font=font)

    scale = min(width / tight_w, height / tight_h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(tight_w * scale))
        new_h = max(1, int(tight_h * scale))
        tight_img = tight_img.resize((new_w, new_h), resample=Image.BILINEAR)
    else:
        new_w, new_h = tight_w, tight_h

    canvas = Image.new("L", (width, height), color=0)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(tight_img, (x, y))

    return np.array(canvas, dtype=np.float32) / 255.0, width


def render_image_template(image_path, height, min_width=MIN_WIDTH, max_width=MAX_WIDTH):
    """Load image -> grayscale -> resize to (width,height) while preserving aspect ratio."""
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if orig_h == 0:
        raise ValueError("Invalid image height.")

    target_w = int(height * (orig_w / orig_h))
    width = max(min_width, min(target_w, max_width))

    img_resized = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    # mild contrast shaping (helps visibility)
    arr = arr ** 0.9

    return arr, width


# ---------- Audio Synthesis ----------
def template_to_hidden_audio(img_arr, n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.5):
    """Template image -> magnitude spectrogram -> audio via Griffin-Lim."""
    mag = np.flipud(img_arr)
    mag = mag ** power
    y = librosa.griffinlim(mag, n_iter=64, hop_length=hop_length, win_length=n_fft)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.9
    return y


# ---------- Saving ----------
def save_audio(y, sr, out_path):
    audio_int16 = (y * 32767).astype(np.int16)
    wav_write(out_path, sr, audio_int16)


def save_audio_spectrogram(y, sr, n_fft, hop_length, out_path, title="Spectrogram"):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="log", cmap="magma"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- Embedding ----------
def embed_in_audio_centered(carrier_path, hidden_y, target_sr=SR, mix_level=EMBED_MIX_LEVEL):
    """Embed hidden_y into carrier audio, centered in time."""
    carrier, sr = librosa.load(carrier_path, sr=target_sr, mono=True)

    carrier_len = len(carrier)
    hidden_len = len(hidden_y)

    if hidden_len > carrier_len:
        hidden_y = hidden_y[:carrier_len]
        hidden_len = carrier_len

    hidden_pad = np.zeros(carrier_len, dtype=np.float32)
    start = (carrier_len - hidden_len) // 2
    hidden_pad[start:start + hidden_len] = hidden_y

    mixed = carrier.astype(np.float32) + mix_level * hidden_pad

    max_amp = np.max(np.abs(mixed))
    if max_amp > 0:
        mixed = mixed / max_amp * 0.9

    return mixed, sr


# ---------- Inspect / Detect ----------
def compute_spectrogram_features(audio_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Heuristic detector for "image-like" / "text-like" spectrogram alterations.
    Returns:
      score (float), details (dict), S_norm (0..1), mask (0/1)
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Normalize to 0..1 for analysis
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)

    # Focus on upper frequency region (embedded art often lives higher; less musical masking)
    # Keep top 60% of frequency bins
    freq_bins = S_norm.shape[0]
    top_start = int(freq_bins * 0.40)
    region = S_norm[top_start:, :]

    # Binary mask: keep strongest energy areas in that region
    # Use percentile threshold to adapt across recordings
    thresh = np.percentile(region, 92)  # tuneable
    mask = (region >= thresh).astype(np.uint8)

    # Features:
    bright_density = float(mask.mean())

    # Projection “structure”: text/images create non-random column/row density variations
    col_profile = mask.mean(axis=0)
    row_profile = mask.mean(axis=1)
    structure_score = float(col_profile.var() + row_profile.var())

    # Simple edge measure: structured shapes raise gradients
    gx = np.abs(np.diff(region, axis=1)).mean() if region.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(region, axis=0)).mean() if region.shape[0] > 1 else 0.0
    edge_score = float(gx + gy)

    # Combine into a suspicion score (heuristic weights)
    score = (2.2 * structure_score) + (1.4 * bright_density) + (0.6 * edge_score)

    details = {
        "threshold_percentile": 92,
        "bright_density": bright_density,
        "structure_score": structure_score,
        "edge_score": edge_score,
        "final_score": float(score),
    }

    return float(score), details, S_norm, mask, top_start


def save_mask_image(mask, out_path):
    # mask is 0/1 -> 0/255
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img.save(out_path)


def run_inspector():
    print("\n🔎 Inspect Mode: Spectrogram Alteration Detector")
    print("-----------------------------------------------")
    audio_path = input("Enter path to audio file to inspect (wav/mp3/etc.): ").strip()
    if not audio_path or not os.path.exists(audio_path):
        print("Audio file not found. Exiting inspect mode.")
        return

    out_spec = input("Enter output spectrogram image name (e.g., inspected_spec.png): ").strip() or "inspected_spec.png"
    base, _ = os.path.splitext(out_spec)
    out_mask = f"{base}_mask.png"

    score, details, S_norm, mask, top_start = compute_spectrogram_features(audio_path)

    # Save pretty spectrogram image
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    save_audio_spectrogram(y, SR, N_FFT, HOP_LENGTH, out_spec, title="Spectrogram")

    # Save mask image (only the analyzed high-frequency region)
    save_mask_image(mask, out_mask)

    # Interpret score (thresholds are heuristic; tune with your own samples)
    # These thresholds work decently for separating “random music/speech” vs “structured embedded art”
    if score >= 0.35:
        verdict = "HIGHLY SUSPICIOUS (likely embedded text/image pattern)"
    elif score >= 0.20:
        verdict = "SUSPICIOUS (possible embedding / unusual structure)"
    else:
        verdict = "LOW SUSPICION (no strong evidence of spectrogram art embedding)"

    print("\n--- Results ---")
    print(f"Verdict: {verdict}")
    print(f"Score: {details['final_score']:.4f}")
    print(f"Bright density: {details['bright_density']:.4f}")
    print(f"Structure score: {details['structure_score']:.6f}")
    print(f"Edge score: {details['edge_score']:.4f}")
    print(f"\nSaved:")
    print(f"  - Spectrogram: {out_spec}")
    print(f"  - Suspicious mask (high-freq region): {out_mask}")
    print("\nTip: If your encoder embeds very strongly, the mask will often reveal letter/image silhouettes.")


# ---------- Main Program ----------
def main():
    print("\n🎵 Spectrogram Stego Tool 🎵")
    print("-----------------------------------")
    print("Choose a mode:")
    print("  1) Text   -> hide text in spectrogram")
    print("  2) Image  -> hide image in spectrogram")
    print("  3) Inspect -> detect likely embedded message/image in an audio spectrogram")

    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "3":
        run_inspector()
        return

    # modes 1 & 2 outputs
    audio_out = input("Enter output WAV file name (e.g., output.wav): ").strip() or "output.wav"
    spec_out = input("Enter output spectrogram image name (e.g., output_spec.png): ").strip() or "output_spec.png"

    height = N_FFT // 2 + 1

    if mode == "1":
        text = input("Enter the text you want to encode: ").strip()
        print("\n[+] Rendering text template...")
        template_arr, used_width = render_text_image(
            text,
            min_width=MIN_WIDTH,
            max_width=MAX_WIDTH,
            height=height,
            font_size=FONT_SIZE,
        )
        template_kind = "text"

    elif mode == "2":
        image_path = input("Enter path to image file (png/jpg/etc.): ").strip()
        if not image_path or not os.path.exists(image_path):
            print("Image file not found. Exiting.")
            return

        print("\n[+] Rendering image template...")
        template_arr, used_width = render_image_template(
            image_path=image_path,
            height=height,
            min_width=MIN_WIDTH,
            max_width=MAX_WIDTH,
        )
        template_kind = "image"

    else:
        print("Invalid choice. Exiting.")
        return

    print(f"[+] Final {template_kind} template size: {used_width} x {height}")

    base, _ = os.path.splitext(spec_out)
    template_path = f"{base}_template.png"
    Image.fromarray((template_arr * 255).astype(np.uint8)).save(template_path)
    print(f"[+] Saved template image: {template_path}")

    print("[+] Generating hidden audio with Griffin-Lim reconstruction...")
    hidden_y = template_to_hidden_audio(template_arr)

    carrier_path = input(
        "\nOptional: Enter path to an existing audio file to embed into "
        "(press Enter to skip and use hidden-only): "
    ).strip()

    if carrier_path:
        if not os.path.exists(carrier_path):
            print(f"Carrier audio not found: {carrier_path}")
            return
        print(f"[+] Embedding hidden {template_kind} into carrier audio (centered): {carrier_path}")
        final_y, final_sr = embed_in_audio_centered(carrier_path, hidden_y, target_sr=SR)
    else:
        print("[+] No carrier provided. Using hidden-only audio.")
        final_y, final_sr = hidden_y, SR

    print(f"[+] Saving final audio: {audio_out}")
    save_audio(final_y, final_sr, audio_out)

    print(f"[+] Saving spectrogram image of final audio: {spec_out}")
    save_audio_spectrogram(final_y, final_sr, N_FFT, HOP_LENGTH, spec_out, title="Spectrogram with Embedded Content")

    print("\n✅ Done! Files created:")
    print(f"   - Final audio: {audio_out}")
    print(f"   - Template image: {template_path}")
    print(f"   - Spectrogram of final audio: {spec_out}")
    print("\nTip: Open the final audio in Audacity / Sonic Visualiser to see the hidden content.")


if __name__ == "__main__":
    main()
