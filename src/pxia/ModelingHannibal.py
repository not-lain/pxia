from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

default_conf = OmegaConf.create({"a": 2, "b": 1})

model_card_template = """
---
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration.

Library: [pxia]({{repo_url}})

## how to load
```
pip install pxia
```

use the AutoModel class 
```python
from pxia AutoModel
model = AutoModel.from_pretrained("{{ repo_id | default("phxia/hannibal", true) }}")
```
or you can use the model class directly
```python
from pxia import Hannibal
model = Hannibal.from_pretrained("{{ repo_id | default("phxia/hannibal", true ) }}")
```

## Contributions
Any contributions are welcome at https://github.com/not-lain/pxia.

<img src="https://huggingface.co/spaces/phxia/README/resolve/main/logo.png"/>

"""


def serialize(x):
    return OmegaConf.to_container(x)


def deserialize(x):
    return OmegaConf.create(x)


class Hannibal(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pxia",
    repo_url="https://github.com/not-lain/pxia",
    tags=["pxia", "hannibal"],
    model_card_template=model_card_template,
    coders={
        DictConfig: (
            lambda x: serialize(x),
            lambda data: deserialize(data),
        )
    },
):
    """an AI model for visual question answering"""

    def __init__(self, cfg: DictConfig = default_conf):
        super().__init__()
        self.cfg = cfg
        self.layer = nn.Linear(cfg.a, cfg.b, bias=False)

    def forward(self, input_ids):
        return self.layer(input_ids)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        model_card_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if model_card_kwargs is None:
            model_card_kwargs = {}
        model_card_kwargs["repo_id"] = repo_id
        return super().push_to_hub(
            repo_id,
            config=config,
            commit_message=commit_message,
            private=private,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            model_card_kwargs=model_card_kwargs,
        )


Hannibal.push_to_hub.__doc__ = PyTorchModelHubMixin.push_to_hub.__doc__
