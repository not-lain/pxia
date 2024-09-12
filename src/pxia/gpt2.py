# Text generation model
import torch
from torch import nn
import math
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional, Tuple

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

    def forward(self, x: torch.LongTensor):
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

    def forward(self, x: torch.LongTensor):
        B, T, C = x.size()  # batch_size, sequence length, embedding_dim (n_dim)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
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
        self,
        block_size: int = 1024,
        vocab_size: int = 50257,
        n_layer: int = 11,
        n_head: int = 12,
        n_embed: int = 768,
    ):
        """
        Initialize the GPT2 model with the given parameters.

        Args:
            block_size (int, optional): The size of the input sequence. Defaults to 1024.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 5025
            n_layer (int, optional): The number of layers in the transformer. Defaults to 11.
            n_head (int, optional): The number of attention heads in each layer. Defaults to 12.
            n_embed (int, optional): The size of the embedding layer. Defaults to 768.

        """

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        This method computes the forward pass of the model.
        It takes as input a tensor of token indices (ids) and computes the logits for the next token.

        Args:
            input_ids (torch.Tensor): A tensor of shape (B, T) containing token indices. B is the batch size


        Returns:
            torch.Tensor: A tensor of shape (B, T, vocab_size) containing the logits for the next token.
        """
        B, T = input_ids.size()
        assert (
            T <= self.block_size
        ), f"cannot forward sequence of length {T}, block_size is {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # positional embedding of shape (T,n_embed)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # taken from original implementation
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        return logits

    @classmethod
    def from_origin(cls, repo_id: str):
        """
        Loads a pretrained model from original huggingface repo, adapts them to the current model and injects them
        This due to mismatch in some layers (we are using Linear instead of Conv1D in the attention layer and other naming mismatches)
        how to use :
        ```python
        >>> model = GPT2LMHeadModel.from_origin('openai-community/gpt2')
        ```

        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            print(
                "Please install transformers library to load model weights from huggingface hub"
            )
            return None
        model_hf = AutoModelForCausalLM.from_pretrained(repo_id)
        # params = inspect.getargspec(cls.__init__)[0]
        params = ["vocab_size", "n_layer", "n_head"]
        kwargs = {k: getattr(model_hf.config, k, None) for k in params}
        model = cls(**kwargs)
        # sanitize state_dict
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
