import os

# Tests must never call the OpenAI API. Profile training would otherwise try
# to summarise interview answers with a real request; tests that cover the
# behavior profile enable it explicitly with a fake client.
os.environ.setdefault("PERSONA_BEHAVIOR_PROFILE_ENABLED", "false")
