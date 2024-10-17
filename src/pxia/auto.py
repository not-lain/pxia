from .gpt2 import GPT2
from .ModelingHannibal import Hannibal
from huggingface_hub import model_info, PyTorchModelHubMixin

from functools import wraps


def set_doc(doc):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = doc
        return wrapper

    return decorator


# Inspect repo parameters and return the appropriate model class


class AutoModel:
    @classmethod
    @set_doc(PyTorchModelHubMixin.from_pretrained.__doc__)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        f"{PyTorchModelHubMixin.from_pretrained.__doc__}"

        tags = model_info(pretrained_model_name_or_path).tags
        if "gpt2" in tags:
            return GPT2.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif "hannibal" in tags:
            return Hannibal.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            raise ValueError("this model is not part of pxia")
