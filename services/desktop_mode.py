import queue
import threading
import tkinter as tk
import unicodedata

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
CHECK_INTERVAL = 1.0


class SofiaDetector:
    def __init__(self):
        print("Loading Whisper...")

        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("Whisper ready.")

        self.audio_queue = queue.Queue()
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.running = True
        self.detecting = False

        self.root = tk.Tk()
        self.root.withdraw()

        self.window = tk.Toplevel(self.root)
        self.window.configure(bg="#ff69b4")
        self.window.overrideredirect(True)
        self.window.geometry("64x64")

        self.window.withdraw()

        self.root.after(100, self.process_audio)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=16000,
            callback=self.audio_callback
        )

        self.stream.start()

        print('Listening... Say "Sofia" to activate the pink window.')

        self.root.mainloop()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        self.audio_queue.put(indata[:, 0].copy())

    def normalize(self, text):
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )
        return text

    def process_audio(self):
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            self.audio_buffer = np.concatenate(
                (self.audio_buffer, chunk)
            )

        max_samples = SAMPLE_RATE * WINDOW_SECONDS

        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]

        if len(self.audio_buffer) >= max_samples and not self.detecting:
            self.detecting = True
            threading.Thread(
                target=self.transcribe_audio,
                daemon=True
            ).start()

        self.root.after(int(CHECK_INTERVAL * 1000), self.process_audio)

    def transcribe_audio(self):
        try:
            audio = self.audio_buffer.copy()

            segments, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=5,
                condition_on_previous_text=False
            )

            text = "".join(segment.text for segment in segments).strip()

            if text:
                print("Heard:", text)

            normalized = self.normalize(text)

            if "sofia" or "sophia" in normalized:
                self.root.after(0, self.show_window)

        except Exception as error:
            print("Transcription error:", error)

        finally:
            self.detecting = False

    def show_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.window.geometry(
            f"64x64+{screen_width - 80}+{screen_height - 100}"
        )
        self.window.deiconify()
        self.window.lift()
        self.root.after(3000, self.window.withdraw)

    def close(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        self.root.destroy()


if __name__ == "__main__":
    SofiaDetector()
