import transformers
import importlib.util

print(transformers.__file__)
print(importlib.util.find_spec("transformers.models.eurobert"))