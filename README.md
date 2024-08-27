# PXIA
A repository for phoenix-ia models 

This repository is using Hugging Face's PyTorchModelHubMixin classes

## How to use

```
pip install pxia
```

```python
from pxia import GPT2
model = GPT2(a=2,b=1)
model.push_to_hub("phxia/gpt2")

pretrained_model = Hannibal.from_pretrained("phxia/gpt2")
```


![pxia](https://github.com/not-lain/pxia/blob/main/logo.png?raw=true)