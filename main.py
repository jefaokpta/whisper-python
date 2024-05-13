import whisper

model = whisper.load_model("large")

filepath = "audios/anderson.m4a"
# filepath = "audios/geladeira.mp3"

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(filepath)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
# options = whisper.DecodingOptions(language="pt", fp16=False)
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print("parcial")
# print(result.text)

result = model.transcribe(filepath, language="pt", fp16=False, beam_size=5, patience=2)

print(result["text"])