from torch import nn
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from mixin import Revised_Mixin

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
model = AutoModel.from_pretrained("{{ repo_id | default("phxia/ann", true) }}")
```
or you can use the model class directly
```python
from pxia import ANN
model = ANN.from_pretrained("{{ repo_id | default("phxia/ann", true ) }}")
```

## Contributions
Any contributions are welcome at https://github.com/not-lain/pxia.

<img src="https://huggingface.co/spaces/phxia/README/resolve/main/logo.png"/>

"""
default_conf = OmegaConf.create({"a": 2, "b": 1})


# we can't store OmegaConf directly in the config.json as init parameter
# so we need to convert it to a dict
def serialize(x):
    return OmegaConf.to_container(x)


def deserialize(x):
    return OmegaConf.create(x)


class ANN(
    nn.Module,
    Revised_Mixin,
    library_name="pxia",
    repo_url="https://github.com/not-lain/pxia",
    tags=["pxia", "ann"],
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


