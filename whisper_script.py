import whisper

model = whisper.load_model("tiny.en")

# Load and trim audio to be under 30 seconds (encodes the audio)
audio = whisper.load_audio("convo_under_30.mp3")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)

# decode audio
options = whisper.DecodingOptions(language=en, fp16=False)
result = whisper.decode(model, mel, options)

print(result.text)
