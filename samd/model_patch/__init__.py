from .llama import llama_patch_dict, llama_attn_patch_dict
from .llama_eagle3 import llama_eagle3_patch_dict, llama_eagle3_attn_patch_dict

patch_dict = {}
attn_patch_dict = {}

patch_dict.update(llama_patch_dict)
attn_patch_dict.update(llama_attn_patch_dict)

eagle3_patch_dict = {}
eagle3_attn_patch_dict = {}

eagle3_patch_dict.update(llama_eagle3_patch_dict)
eagle3_attn_patch_dict.update(llama_eagle3_attn_patch_dict)
