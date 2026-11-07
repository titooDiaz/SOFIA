from flask import Flask, render_template, request, send_file, jsonify
from dotenv import load_dotenv
from services.brain import SofiaBrain
from services.listener import SofiaListener
from services.voice import SofiaVoice
from services.commander import DeviceCommander 
from services.desktop_mode import DesktopMode
from agent.desktop_agent import DesktopAgent

import os

load_dotenv()

app = Flask(__name__)

print("Starting SOFIA...")

brain = SofiaBrain()
listener = SofiaListener()
voice = SofiaVoice()
commander = DeviceCommander()
desktop_mode = None
desktop_agent = DesktopAgent(model=os.getenv("OLLAMA_AGENT_MODEL", DesktopAgent.MODEL))

print("SOFIA is ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    audio = request.files["audio"]
    manipular_dispositivo = request.form.get("manipulate") == "true"

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
    # Device Manipulation Flow
    # ---------------------
    if manipular_dispositivo:
        prompt_decision = f"answer 'YES' or 'NO' to the following question: Should I execute the command '{text}'?"
        decision = brain.think(prompt_decision).strip().upper()
        
        if "YES" in decision:
            print("SOFIA: Intentando ejecutar comando...")
            
            # This triggers the loop in commander.py until success or max retries
            success, message = commander.execute_intent(text)
            
            print(f"SOFIA Result: {message}")
            
            return jsonify({
                "status": "OK" if success else "ERROR", 
                "message": message
            })

    # ---------------------
    # Conversational Flow (LLM)
    # ---------------------
    response = brain.think(text)
    print(f"SOFIA: {response}")

    if not response:
        return {"error": "SOFIA generated an empty response"}, 500

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


@app.route("/agent/run", methods=["POST"])
def run_agent():
    data = request.get_json(silent=True) or {}
    goal = data.get("goal", "")
    if not isinstance(goal, str):
        return jsonify({"error": "La tarea debe ser texto."}), 400
    try:
        return jsonify(desktop_agent.run(goal))
    except (RuntimeError, ValueError) as error:
        return jsonify({"error": str(error)}), 409


@app.route("/agent/stop", methods=["POST"])
def stop_agent():
    desktop_agent.stop()
    return jsonify({"status": "stopping"})

if __name__ == "__main__":
    desktop_enabled = os.getenv("Desktopmode", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if True:
        desktop_mode = DesktopMode(listener, brain, voice, commander)
        desktop_mode.start()

    app.run(
        host="127.0.0.1",
        port=3000,
        debug=False,
        use_reloader=False
    )