# PXIA
A repository for pxia models 

This repository is using HuggingFace's PyTorchModelHubMixin classes

[![Downloads](https://static.pepy.tech/badge/pxia)](https://pepy.tech/project/pxia)

## How to use

```
pip install pxia
```

```python
from pxia import GPT2
model = GPT2(block_size= 1024, vocab_size = 50257, n_layer= 11, n_head= 12, n_embed = 768) # or use default parameters
model.push_to_hub("phxia/gpt2")

pretrained_model = GPT2.from_pretrained("phxia/gpt2")
```

alternatively you can load weights from source gpt2 models from huggingface and convert them to pxia format

```python
model = GPT2.from_origin("openai-community/gpt2")
```

![pxia](https://github.com/not-lain/pxia/blob/main/logo.png?raw=true)