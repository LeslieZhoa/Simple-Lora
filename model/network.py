TEXT_ENCODER_ATTN_MODULE = ".self_attn"
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
import torch.nn.functional as F
class Lora:
    lora_scale  = 1.
    def get_unet_lora_layer(self):
        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )
            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet.set_attn_processor(unet_lora_attn_procs)
        return AttnProcsLayers(self.unet.attn_processors).to(self.device)
    
    def get_text_encoder_lora_layer(self):
        text_lora_attn_procs = {}
        for name, module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                text_lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=module.out_proj.out_features, cross_attention_dim=None
                )
        text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs).to(self.device)
        self._modify_text_encoder(text_lora_attn_procs)
        return text_encoder_lora_layers

    
    def _remove_text_encoder_monkey_patch(self):
        # Loop over the CLIPAttention module of text_encoder
        for name, attn_module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for _, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of CLIPAttention
                    module = attn_module.get_submodule(text_encoder_attr)
                    if hasattr(module, "old_forward"):
                        # restore original `forward` to remove monkey-patch
                        module.forward = module.old_forward
                        delattr(module, "old_forward")
    def _modify_text_encoder(self,attn_processors):
        self._remove_text_encoder_monkey_patch()
        # Loop over the CLIPAttention module of text_encoder
        
        for name, attn_module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for attn_proc_attr, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of CLIPAttention and its corresponding LoRA layer.
                    module = attn_module.get_submodule(text_encoder_attr)
                    lora_layer = attn_processors[name].get_submodule(attn_proc_attr)

                    # save old_forward to module that can be used to remove monkey-patch
                    old_forward = module.old_forward = module.forward

                    # create a new scope that locks in the old_forward, lora_layer value for each new_forward function
                    # for more detail, see https://github.com/huggingface/diffusers/pull/3490#issuecomment-1555059060
                    def make_new_forward(old_forward, lora_layer):
                        def new_forward(x):
                            result = old_forward(x) + self.lora_scale * lora_layer(x)
                            return result

                        return new_forward

                    # Monkey-patch.
                    module.forward = make_new_forward(old_forward, lora_layer)

    @property
    def _lora_attn_processor_attr_to_text_encoder_attr(self):
        return {
            "to_q_lora": "q_proj",
            "to_k_lora": "k_proj",
            "to_v_lora": "v_proj",
            "to_out_lora": "out_proj",
        }
