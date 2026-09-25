from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import numpy as np

model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate(
    text="The great big monkey ate all the banannas!",
    exaggeration=0.5,
)

# wav comes back as a torch tensor - convert to numpy for soundfile
wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
sf.write("test_output.wav", wav_np, model.sr)
print("Done! Saved to test_output.wav")
