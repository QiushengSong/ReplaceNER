import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch.nn as nn
from typing import Union
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file


class MambaForNER(nn.Module):
    r"""NER model class

    class MambaConfig:
        - d_model: int = 2560
        - n_layer: int = 64
        - vocab_size: int = 50277
        - ssm_cfg: dict = field(default_factory=dict)
        - rms_norm: bool = True
        - residual_in_fp32: bool = True
        - fused_add_norm: bool = True
        - pad_vocab_size_multiple: int = 8
        - tie_embeddings: bool = True

    Args:
        config: MambaConfig
        num_class: The number of class of entity
        device: Device
        dtype: Data type

    **Attributes**:
        - **self.mamba_config** (class) -- A configuration class including some attributes about mamba model.
        - **self.backbone** -- MambaLMHeadModel


    """
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        num_class: int = None,
        device=None,
        dtype=torch.float16
    ) -> None:

        super().__init__()
        self.mamba_config = config
        self.backbone = MambaLMHeadModel(self.mamba_config,
                                         initializer_cfg=initializer_cfg,
                                         device=device,
                                         dtype=dtype)

        self.backbone.lm_head = nn.Linear(in_features=self.mamba_config.d_model,
                                          out_features=num_class, bias=False, dtype=dtype)

        weight_file = cached_file('state-spaces/mamba-2.8b', WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        state_dict = torch.load(weight_file)
        del state_dict['lm_head.weight']
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):

        output = self.backbone(input_ids,
                               position_ids=position_ids,
                               inference_params=inference_params,
                               num_last_tokens=num_last_tokens)

        return output


class PTMambaNER(nn.Module):
    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            # prefix_tuning: bool = False,
            # virtual_tokens: int = 30,
            device=None,
            dtype=torch.float32
            ) -> None:

        super().__init__()
        self.mamba_config = config
        # self.prefix_tuning = prefix_tuning
        self.backbone = MambaLMHeadModel(self.mamba_config,
                                         initializer_cfg=initializer_cfg,
                                         device=device,
                                         dtype=dtype)

        # weight_file = cached_file('state-spaces/mamba-2.8b', WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        state_dict = torch.load("mamba-2.8b.pt")
        self.backbone.load_state_dict(state_dict, strict=True)
        for name, param in self.backbone.named_parameters():
            if name == 'backbone.embedding.weight':
                param.requires_grad = False
        # if self.prefix_tuning:
        #     self.prefix_prompt_encoder = nn.Embedding(virtual_tokens, self.mamba_config.d_model)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        output = self.backbone(input_ids,
                               position_ids=position_ids,
                               inference_params=inference_params,
                               num_last_tokens=num_last_tokens)

        return output


class TestModel(nn.Module):
    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            # prefix_tuning: bool = False,
            # virtual_tokens: int = 30,
            device=None,
            dtype=torch.float32
            ) -> None:

        super().__init__()
        self.mamba_config = config
        # self.prefix_tuning = prefix_tuning
        self.backbone = MambaLMHeadModel(self.mamba_config,
                                         initializer_cfg=initializer_cfg,
                                         device=device,
                                         dtype=dtype)

        # weight_file = cached_file('state-spaces/mamba-130m', WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        state_dict = torch.load("mamba-130m.pt")
        self.backbone.load_state_dict(state_dict, strict=True)
        for name, param in self.backbone.named_parameters():
            if name == 'backbone.embedding.weight':
                param.requires_grad = False
        # if self.prefix_tuning:
        #     self.prefix_prompt_encoder = nn.Embedding(virtual_tokens, self.mamba_config.d_model)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        output = self.backbone(input_ids,
                               position_ids=position_ids,
                               inference_params=inference_params,
                               num_last_tokens=num_last_tokens)

        return output



