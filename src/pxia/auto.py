from .gpt2 import GPT2
from .ModelingHannibal import Hannibal
from huggingface_hub import model_info , PyTorchModelHubMixin

# Inspect repo parameters and return the appropriate model class

class AutoModel:
    @classmethod
    def from_pretrained(cls, repo_id, *model_args, **kwargs):
        tags = model_info(repo_id).tags
        if "gpt2" in tags:
            return GPT2.from_pretrained(repo_id,*model_args,**kwargs)
        elif "hannibal" in tags:
            return Hannibal.from_pretrained(repo_id,*model_args,**kwargs)
        else:
            raise ValueError("this model is not part of pxia")

AutoModel.from_pretrained.__doc__ = PyTorchModelHubMixin.from_pretrained.__doc__