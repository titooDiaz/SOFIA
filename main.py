from flask import Flask, render_template, request, send_file, jsonify
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
    manipular_activo = request.form.get("manipulate") == "true"

    input_file = "input.wav"
    audio.save(input_file)

    # ---------------------
    # Speech -> Text
    # ---------------------
    text = listener.transcribe(input_file)
    print(f"You: {text}")

    if not text:
        return {"error": "No speech detected"}, 400

    # ---------------------
    # Desition: Is it a command for the PC?
    # ---------------------
    if manipular_activo:
        # Prompt to determine if the user input is a command to interact with the computer
        prompt_decision = (
            f"The user said: '{text}'. "
            "Does this text imply a command to interact or manipulate the computer "
            "(such as opening programs, creating files, shutting down, etc)? "
            "Respond ONLY with the word 'YES' or the word 'NO'."
        )
        decision = brain.think(prompt_decision).strip().upper()
        
        # if sofia decides it's a command to interact with the PC
        if "YES" in decision:
            print("OK")
            return jsonify({"status": "command_executed", "message": "OK"})

    # ---------------------
    # LLM
    # ---------------------
    response = brain.think(text)
    print(f"SOFIA: {response}")

    if not response:
        return {"error": "SOFIA generated an empty response"}, 500

    audio_file = voice.generate(response)
    print(f"Audio file: {audio_file}")

    if not audio_file:
        return {"error": "Voice generation failed", "response": response}, 500

    return send_file(
        audio_file,
        mimetype="audio/wav"
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=666, debug=True)