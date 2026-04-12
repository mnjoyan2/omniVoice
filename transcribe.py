import sherpa_onnx
import os
import sys

# 1. Path Configuration
model_dir = "./asr_models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
tokens_path = os.path.join(model_dir, "tokens.txt")
model_path = os.path.join(model_dir, "model.int8.onnx")

# 2. Setup Configuration
# We use the specific NeMo CTC config for your int8 model
model_config = sherpa_onnx.OfflineModelConfig(
    nemo_ctc=sherpa_onnx.OfflineNemoEncDecCtcModelConfig(model=model_path),
    tokens=tokens_path,
    num_threads=4,
    debug=False
)

recon_config = sherpa_onnx.OfflineRecognizerConfig(
    model_config=model_config,
    decoding_method="greedy_search"
)

# 3. The Factory Initialization (The fix)
# In the latest API, we use .create() instead of the constructor
recognizer = sherpa_onnx.OfflineRecognizer.create(config=recon_config)

# 4. Processing Logic
test_file = sys.argv[1] if len(sys.argv) > 1 else ""

if os.path.exists(test_file):
    print(f"--- Transcribing Armenian: {os.path.basename(test_file)} ---")
    
    # Read wave
    samples, sample_rate = sherpa_onnx.read_wave(test_file)
    
    # Create stream, accept audio, and decode
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    recognizer.decode_stream(stream)
    
    print(f"\nRESULT:\n{stream.result.text.strip()}\n")
else:
    print("Error: Please provide a valid path to an Armenian .wav file.")
