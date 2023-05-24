import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name: str, description: str):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def extract_code(text):
    texts = text.split("```")
    code = None
    if len(texts) > 1:
        code = texts[1]
        code = code.strip("python").strip("Python")
    return code