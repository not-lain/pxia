# PXIA
A repository for phoenix-ia models 
This repository is using Hugging Face's PyTorchModelHubMixin classes

## How to use

```
pip install pxia
```

```python
from pxia import Hannibal
model = Hannibal(a=2,b=1)
model.push_to_hub("phxia/Hannibal")

pretrained_model = Hannibal.from_pretrained("phxia/Hannibal")
```


![pheonixia](https://github.com/not-lain/pxia/blob/main/logo.png?raw=true)