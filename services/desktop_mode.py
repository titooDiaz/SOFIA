import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write


class DesktopIndicator:

    _window_script = r'''
import queue
import sys
import tkinter as tk
import threading

commands = queue.Queue()

def read_commands():
    for line in sys.stdin:
        commands.put(line.strip())

root = tk.Tk()
root.withdraw()
window = tk.Toplevel(root)
window.overrideredirect(True)
window.attributes("-topmost", True)
window.configure(bg="#ff5cab")
height = root.winfo_screenheight()
window.geometry(f"24x24+24+{height - 48}")
window.withdraw()

def poll():
    try:
        while True:
            command = commands.get_nowait()
            if command == "show":
                window.deiconify()
            elif command == "hide":
                window.withdraw()
    except queue.Empty:
        pass
    root.after(50, poll)

threading.Thread(target=read_commands, daemon=True).start()
poll()
root.mainloop()
'''

    def __init__(self):
        try:
            self.process = subprocess.Popen(
                ["python3", "-c", self._window_script],
                stdin=subprocess.PIPE,
                text=True
            )
        except OSError as error:
            print(f"Desktop indicator unavailable: {error}")

    def set_active(self, active):
        if not hasattr(self, "process") or self.process.poll() is not None:
            return
        try:
            self.process.stdin.write("show\n" if active else "hide\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass


class DesktopMode:

    sample_rate = 16000
    wake_window_seconds = 2.0
    speech_chunk_seconds = 0.25
    silence_seconds = 0.9
    silence_threshold = 0.018
    max_speech_seconds = 15

    def __init__(self, listener, brain, voice, commander):
        self.listener = listener
        self.brain = brain
        self.voice = voice
        self.commander = commander
        self.running = threading.Event()
        self.indicator = DesktopIndicator()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        print("Desktop mode enabled. Say 'sofia' to activate.")
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.indicator.set_active(False)

    def _record(self, seconds):
        audio = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16"
        )
        sd.wait()
        return audio

    def _transcribe_audio(self, audio):
        filename = tempfile.mktemp(suffix=".wav")
        try:
            write(filename, self.sample_rate, audio)
            return self.listener.transcribe(filename)
        finally:
            try:
                os.remove(filename)
            except OSError:
                pass

    def _record_command(self):
        chunks = []
        silence_started = None
        started = time.monotonic()

        while self.running.is_set():
            chunk = self._record(self.speech_chunk_seconds)
            chunks.append(chunk)
            level = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768))))
            now = time.monotonic()

            if level < self.silence_threshold:
                silence_started = silence_started or now
                if now - silence_started >= self.silence_seconds:
                    break
            else:
                silence_started = None

            if now - started >= self.max_speech_seconds:
                break

        if not chunks:
            return ""
        return self._transcribe_audio(np.concatenate(chunks))

    def _process(self, text):
        print(f"You: {text}")
        prompt_decision = (
            f"answer 'YES' or 'NO' to the following question: "
            f"Should I execute the command '{text}'?"
        )
        decision = self.brain.think(prompt_decision).strip().upper()

        if "YES" in decision:
            success, message = self.commander.execute_intent(text)
            print(f"SOFIA Result: {message}")
            return

        response = self.brain.think(text)
        print(f"SOFIA: {response}")
        audio_file = self.voice.generate(response)
        if not audio_file:
            return

        audio, sample_rate = sf.read(audio_file, dtype="float32")
        sd.play(audio, sample_rate)
        sd.wait()
        try:
            os.remove(audio_file)
        except OSError:
            pass

    def _run(self):
        while self.running.is_set():
            try:
                wake_audio = self._record(self.wake_window_seconds)
                wake_text = self._transcribe_audio(wake_audio)
                if "sofia" not in wake_text.lower():
                    continue

                self.indicator.set_active(True)
                text = self._record_command()
                if text:
                    self._process(text)
            except Exception as error:
                print(f"Desktop mode error: {error}")
                time.sleep(1)
            finally:
                self.indicator.set_active(False)