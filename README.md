# LLM Social Simulation Generalization

Code for replicating five published LLM-based social simulation paradigms across contemporary language models: the behavioural Turing test (`LLM-Behavioral`), simulated altruism (`LLM-Altruism`), identity representation (`LLM-Can-Harmfully-Misportray-and-Flatten-Identity-Groups`), age and gender bias in resumes (`LLM-Age-Gender-Distortion`) and convention formation in the naming game (`LLM-Social-Conventions-and-Collective-Bias`). Each directory has a `run.py` entry point and its own configuration file; `orchestrator` provides a web interface that runs all paradigms through one OpenAI-compatible endpoint.

Install the dependencies with Python 3.11 using `pip install -r requirements.txt`. Set `LLM_BASE_URL` and `LLM_API_KEY` for your model endpoint, then start the orchestrator with `PY=$(which python) ./orchestrator/dev-all.sh`, or run a single paradigm with `python <directory>/run.py`.
