# Text generation model
import torch
from torch import nn
import math
from torch.nn import functional as F
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
from pxia import GPT2
model = GPT2.from_pretrained("{{ repo_id | default("phxia/GPT2") }}")
```

## Contributions
Any contributions are welcome at https://github.com/not-lain/pxia 

## Myth
A phoenix is a legendary creature that was part of the ancient Phoenician empire, known for its barbarian warriors who with their general Hannibal fought against the Roman empire.

<img src="https://huggingface.co/spaces/phxia/README/resolve/main/logo.png"/>

"""


# huggingface/transformers implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py


class MLP(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.c_proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        x = F.gelu(self.c_fc(x), approximate="tanh")
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embed, block_size) -> None:
        super().__init__()
        self.n_embed = n_embed
        assert (
            n_embed % n_head == 0
        ), f"n_head = {n_head} is not divisable by n_embed = {n_embed}"
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch_size, sequence length, embedding_dim (n_dim)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, split_size=C)
        # nh = number of heads, hs = head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        return out


class Block(nn.Module):
    def __init__(self, n_head, n_embed, block_size) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_head, n_embed, block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pxia",
    repo_url="https://github.com/not-lain/pxia",
    tags=["text-generation", "pxia", "hannibal"],
    model_card_template=model_card_template,
):
    """an AI model for visual question answering"""

    def __init__(
        self, block_size=1024, vocab_size=50257, n_layer=11, n_head=12, n_embed=768
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embed),
                wpe=nn.Embedding(block_size, n_embed),
                h=nn.ModuleList(
                    [Block(n_head, n_embed, block_size) for _ in range(n_layer)]
                ),
                ln_f=nn.LayerNorm(n_embed),
            )
        )
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor):
        B, T = ids.size()
        assert (
            T <= self.block_size
        ), f"cannot forward sequence of length {T}, block_size is {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=ids.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # positional embedding of shape (T,n_embed)
        tok_emb = self.transformer.wte(ids)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
