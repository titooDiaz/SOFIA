from flask import Flask, render_template, request, send_file
from services.brain import SofiaBrain
from services.listener import SofiaListener
from services.voice import SofiaVoice

import os


app = Flask(__name__)


print("Starting SOFIA...")

brain = SofiaBrain()
listener = SofiaListener()
voice = SofiaVoice()

print("SOFIA is ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    audio = request.files["audio"]

    input_file = "input.wav"

    audio.save(input_file)

    # ---------------------
    # Speech -> Text
    # ---------------------

    text = listener.transcribe(input_file)

    print(f"You: {text}")

    if not text:
        return {
            "error": "No speech detected"
        }, 400

    # ---------------------
    # LLM
    # ---------------------

    response = brain.think(text)

    print(f"SOFIA: {response}")

    if not response:
        return {
            "error": "SOFIA generated an empty response"
        }, 500

    audio_file = voice.generate(response)

    print(f"Audio file: {audio_file}")

    if not audio_file:
        return {
            "error": "Voice generation failed",
            "response": response
        }, 500

    return send_file(
        audio_file,
        mimetype="audio/wav"
    )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=666,
        debug=True
    )