from torch import nn
from huggingface_hub import PyTorchModelHubMixin


model_card_template = """
---
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration.

Library: [Phoenix-IA]({{repo_url}})

## how to load
```
pip install phoenixia
```

```python
from phoenixia import Hannibal
model = Hannibal.from_pretrained("{{ repo_id | default("phoenix-ia/Hannibal") }}")
```

## Contributions
Any contributions are welcome at https://github.com/not-lain/phoenixia.

## Myth
A phoenix is a legendary creature that was part of the ancient Phoenician empire, known for its barbarian warriors who with their general Hannibal fought against the Roman empire.

The Phoenician empire perished and Tunis-IA rose from its ashes.

<img src="https://huggingface.co/spaces/phoenix-ia/README/resolve/main/logo.png"/>

"""


class Hannibal(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="phoenixia",
    repo_url="https://github.com/not-lain/phoenixia",
    tags=["visual-question-answering", "phoenixia","hannibal"],
    model_card_template=model_card_template,
):
    """an AI model for visual question answering"""

    def __init__(self, a=2, b=1):
        super().__init__()
        self.layer = nn.Linear(a, b, bias=False)

    def forward(self, input_ids):
        return self.layer(input_ids)
