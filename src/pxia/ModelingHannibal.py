from torch import nn
from huggingface_hub import PyTorchModelHubMixin


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

```python
from pxia import Hannibal
model = Hannibal.from_pretrained("{{ repo_id | default("phxia/Hannibal") }}")
```

## Contributions
Any contributions are welcome at https://github.com/not-lain/pxia.

## Myth
A phoenix is a legendary creature that was part of the ancient Phoenician empire, known for its barbarian warriors who with their general Hannibal fought against the Roman empire.

<img src="https://huggingface.co/spaces/phxia/README/resolve/main/logo.png"/>

"""


class Hannibal(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pxia",
    repo_url="https://github.com/not-lain/pxia",
    tags=["visual-question-answering", "pxia", "hannibal"],
    model_card_template=model_card_template,
):
    """an AI model for visual question answering"""

    def __init__(self, a=2, b=1):
        super().__init__()
        self.layer = nn.Linear(a, b, bias=False)

    def forward(self, input_ids):
        return self.layer(input_ids)
