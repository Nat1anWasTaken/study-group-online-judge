from types import MethodType

import torch
from cut_cross_entropy import linear_cross_entropy
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

CCE_REVISION = "3de376c106a1916bc5e1b619f9c77c87a461ee1c"
CCE_IMPLEMENTATION = "cce_exact"


@torch.compiler.disable
def cce_loss(hidden, weight, labels, num_items):
    loss = linear_cross_entropy(
        hidden, weight, labels, shift=1,
        reduction="sum" if num_items is not None else "mean",
        impl=CCE_IMPLEMENTATION,
    )
    return loss if num_items is None else loss / num_items


def cce_forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
    if labels is None:
        return GPT2LMHeadModel.forward(
            self, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, **kwargs,
        )
    num_items = kwargs.pop("num_items_in_batch", None)
    outputs = self.transformer(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=False, return_dict=True,
    )
    hidden = outputs.last_hidden_state
    if torch.is_autocast_enabled("cuda"):
        hidden = hidden.to(torch.get_autocast_dtype("cuda"))
    loss = cce_loss(hidden, self.lm_head.weight.to(hidden.dtype), labels, num_items)
    return CausalLMOutputWithCrossAttentions(loss=loss)


def use_cce(model):
    model.forward = MethodType(cce_forward, model)
    return model
