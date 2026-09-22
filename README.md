### Documents folder

Create a `documents` folder in the project's root directory. Inside this folder, create a file named `commands.json`.

The project structure should look like this:

```text
documents/
    commands.json
    ...
services/
    ...
templates/
    ...
test/
    ...
```

SOFIA is a personal assistant that uses **Ollama models** as its reasoning engine. Therefore, Ollama must be installed and configured on the computer where SOFIA is running.

## Requirements

* Python 3.10+
* Ollama
* An Ollama-compatible model
* The project's Python dependencies

## Ollama Setup

First, install Ollama on your computer and download a model:

```bash
ollama pull qwen3:0.6b
```

SOFIA uses the model configured in `commander.py` and other components. **The recommended model depends on the hardware of each computer.**

For example:

* Low-resource computers -> smaller models such as `qwen3:0.6b`.
* Computers with more RAM, CPU power, or a dedicated GPU → larger models can be used for better reasoning capabilities.

You can change the model directly in the project configuration:

```python
model="qwen3:0.6b"
```

Choose a model based on the available hardware, balancing **speed and reasoning capability**.

## Running SOFIA

Install the project dependencies and run:

```bash
python main.py
```

SOFIA communicates with Ollama locally to process requests.
