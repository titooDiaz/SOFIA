from kokoro import KPipeline
import soundfile as sf
import subprocess

# pip install -U pip
# pip install "kokoro>=0.9.4" soundfile
# sudo apt install espeak-ng

print("Loading SOFIA TTS...")

pipeline = KPipeline(lang_code="e")

print("Listo.")

text = input("Texto para SOFIA: ")

generator = pipeline(
    text,
    voice="ef_dora"
)

for i, (gs, ps, audio) in enumerate(generator):
    filename = f"sofia_{i}.wav"

    sf.write(
        filename,
        audio,
        24000
    )

    subprocess.run(
        ["aplay", filename],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("Terminado.")