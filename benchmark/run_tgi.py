
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engines.tgi_backend import TGIWrapper



engine = TGIWrapper()

responses = engine.generate(["What is the capital of France?", "Explain gravity"])
print(responses)