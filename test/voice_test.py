import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


# Settings
SAMPLE_RATE = 16000
DURATION = 5

print("Loading Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper listo.")


# Record
print("\nSpeak during 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

write(
    "audio.wav",
    SAMPLE_RATE,
    audio
)

print("Recording finished.")


# Write
segments, info = model.transcribe(
    "audio.wav",
    language="es"
)

text = ""

for segment in segments:
    text += segment.text

print("\nSOFIA heard:")
print(text)