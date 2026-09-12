from faster_whisper import WhisperModel


class SofiaListener:

    def __init__(self):

        print("Loading Whisper...")

        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("Whisper ready.")

    def transcribe(self, filename):

        segments, _ = self.model.transcribe(
            filename,
            language="en"
        )

        text = ""

        for segment in segments:
            text += segment.text

        return text.strip()