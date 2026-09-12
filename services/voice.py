from kokoro import KPipeline
import soundfile as sf
import tempfile


class SofiaVoice:

    def __init__(self):

        print("Loading SOFIA voice...")

        self.pipeline = KPipeline(
            lang_code="a"
        )

        self.voice = "af_heart"

        print("SOFIA voice ready.")

    def clean_text(self, text):

        # Saltos de línea → espacios
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

        # Evitar espacios duplicados
        text = " ".join(text.split())

        return text.strip()

    def generate(self, text):

        text = self.clean_text(text)

        generator = self.pipeline(
            text,
            voice=self.voice
        )

        audio_parts = []

        for _, _, audio in generator:
            audio_parts.append(audio)

        if not audio_parts:
            return None

        # Unir TODOS los fragmentos
        import numpy as np

        audio = np.concatenate(audio_parts)

        filename = tempfile.mktemp(
            suffix=".wav"
        )

        sf.write(
            filename,
            audio,
            24000
        )

        return filename