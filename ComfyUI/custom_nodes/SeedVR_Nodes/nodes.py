import torch
import os
from omegaconf import OmegaConf, DictConfig, open_dict # Added open_dict

# ComfyUI imports
import folder_paths
import comfy.model_management as model_management # For device and dtype
import comfy.utils # For progress bar if needed later

# SeedVR imports
from .core.config_utils import load_config, create_object
# We need to ensure that the 'core' module and its submodules are correctly importable.
# This typically means having __init__.py files in 'core', 'core/dit_v2_models', 'core/video_vae_v3_models' etc.
# and that the paths in the configs are adjusted to be relative to the 'core' package or a known root.

class SeedVRModelLoader:
    CATEGORY = "SeedVR"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_model_name": (folder_paths.get_filename_list("checkpoints"), ),
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "seedvr_config_file": (["seedvr_3b_main.yaml"], ), # Default config
            }
        }

    RETURN_TYPES = ("SEEDVR_MODEL", "VAE")
    FUNCTION = "load_models"

    def _adjust_config_paths(self, config_node: DictConfig):
        """
        Recursively adjust __object__.path in the config to be relative to the
        SeedVR_Nodes.core package, assuming create_object's import_item resolves from there.
        e.g., models.dit_v2.nadit -> .dit_v2_models.nadit
              models.video_vae_v3.modules.video_vae.VideoAutoencoderKL -> .video_vae_v3_models.modules.video_vae.VideoAutoencoderKL
        """
        if isinstance(config_node, DictConfig):
            if "__object__" in config_node and "path" in config_node.__object__:
                original_path = config_node.__object__.path
                if original_path.startswith("models.dit_v2"):
                    new_path = original_path.replace("models.dit_v2", ".dit_v2_models", 1)
                    config_node.__object__.path = new_path
                    # print(f"[SeedVRModelLoader] Adjusted DiT path in config: {original_path} -> {new_path}")
                elif original_path.startswith("models.video_vae_v3"):
                    new_path = original_path.replace("models.video_vae_v3", ".video_vae_v3_models", 1)
                    config_node.__object__.path = new_path
                    # print(f"[SeedVRModelLoader] Adjusted VAE path in config: {original_path} -> {new_path}")
                # Add more rules if other top-level model paths exist

            for key, value in config_node.items():
                self._adjust_config_paths(value)
        elif isinstance(config_node, list): # Check for ListConfig as well
            for item in config_node:
                self._adjust_config_paths(item)


    def load_models(self, dit_model_name: str, vae_name: str, seedvr_config_file: str):
        print(f"[SeedVRModelLoader] Starting model loading...")
        device = model_management.get_torch_device()
        # dtype = model_management.VAE_DTYPE # Or choose an appropriate dtype, DiT might use float32

        # 1. Get full paths to model checkpoints
        dit_checkpoint_path = folder_paths.get_full_path("checkpoints", dit_model_name)
        vae_checkpoint_path = folder_paths.get_full_path("vae", vae_name)

        if not dit_checkpoint_path or not os.path.exists(dit_checkpoint_path):
            raise FileNotFoundError(f"DiT checkpoint not found: {dit_model_name} (resolved to {dit_checkpoint_path})")
        if not vae_checkpoint_path or not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_name} (resolved to {vae_checkpoint_path})")

        print(f"[SeedVRModelLoader] DiT Checkpoint: {dit_checkpoint_path}")
        print(f"[SeedVRModelLoader] VAE Checkpoint: {vae_checkpoint_path}")

        # 2. Load OmegaConf configuration
        base_path = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_path, "configs", seedvr_config_file)

        if not os.path.exists(config_path):
            alt_config_path = os.path.join(os.path.dirname(__file__), "configs", seedvr_config_file) # Relative to this nodes.py file
            if os.path.exists(alt_config_path): # This path is less likely to be correct if __file__ is not where we think
                config_path = alt_config_path
            else: # Final attempt: relative to ComfyUI root then custom_nodes path
                # This needs folder_paths.base_path or similar if available, or assume relative to CWD
                # For now, stick to paths relative to this custom node
                 raise FileNotFoundError(f"SeedVR config file not found: {seedvr_config_file} (tried {config_path} and {alt_config_path})")

        print(f"[SeedVRModelLoader] Loading SeedVR config: {config_path}")
        try:
            # The load_config from config_utils.py is in core/, so its relative path assumptions for __inherit__
            # need to be based from core/.
            # If config_path is absolute, OmegaConf should handle it.
            # If __inherit__ uses relative paths, they are relative to the *loaded file's directory*.
            # So, if configs/seedvr_3b_main.yaml has __inherit__: ../core/video_vae_v3_models/s8_c16_t4_inflation_sd3.yaml
            # this should resolve correctly. The original `models/video_vae_v3/...` would not.
            # It's best if the YAML files in `configs` are adjusted to use correct relative paths to items in `core`.
            cfg = load_config(config_path)

            with open_dict(cfg):
                self._adjust_config_paths(cfg)
        except Exception as e:
            print(f"[SeedVRModelLoader] Error loading or adjusting SeedVR config: {e}")
            raise

        # 3. Instantiate DiT model
        print(f"[SeedVRModelLoader] Instantiating DiT model...")
        try:
            dit_model_config = cfg.dit.model
            dit_model = create_object(dit_model_config)
            # dit_model.to(device) # Move to device after loading state_dict if possible to save memory during load

            print(f"[SeedVRModelLoader] Loading DiT state_dict from: {dit_checkpoint_path}")
            state_dict_dit = torch.load(dit_checkpoint_path, map_location="cpu") # Removed mmap=True for broader compatibility

            # Handle nested state dicts (common in training checkpoints)
            if "model" in state_dict_dit:
                state_dict_dit = state_dict_dit["model"]
            elif "state_dict" in state_dict_dit:
                state_dict_dit = state_dict_dit["state_dict"]

            missing_keys, unexpected_keys = dit_model.load_state_dict(state_dict_dit, strict=False)
            if missing_keys: print(f"[SeedVRModelLoader] DiT missing_keys: {missing_keys}")
            if unexpected_keys: print(f"[SeedVRModelLoader] DiT unexpected_keys: {unexpected_keys}")

            dit_model.eval()
            dit_model.to(device)
        except Exception as e:
            print(f"[SeedVRModelLoader] Error loading DiT model: {e}")
            raise

        # 4. Instantiate VAE model
        print(f"[SeedVRModelLoader] Instantiating VAE model...")
        try:
            vae_model_config = cfg.vae.model
            vae = create_object(vae_model_config)
            # vae.to(device) # Move after loading

            print(f"[SeedVRModelLoader] Loading VAE state_dict from: {vae_checkpoint_path}")
            state_dict_vae = torch.load(vae_checkpoint_path, map_location="cpu")
            if "model" in state_dict_vae:
                state_dict_vae = state_dict_vae["model"]
            elif "state_dict" in state_dict_vae:
                state_dict_vae = state_dict_vae["state_dict"]

            missing_keys, unexpected_keys = vae.load_state_dict(state_dict_vae, strict=False)
            if missing_keys: print(f"[SeedVRModelLoader] VAE missing_keys: {missing_keys}")
            if unexpected_keys: print(f"[SeedVRModelLoader] VAE unexpected_keys: {unexpected_keys}")

            vae.eval()
            vae.to(device)
            vae.dtype = model_management.VAE_DTYPE # Ensure VAE has dtype attribute Comfy expects
            # vae.first_stage_model = vae # If VAE is direct, not nested in DDPM like LDM
        except Exception as e:
            print(f"[SeedVRModelLoader] Error loading VAE model: {e}")
            raise

        seedvr_model_data = {
            "dit_model": dit_model,
            "model_config": cfg.dit.model,
            "full_config": cfg,
            # "text_encoder": None, # Placeholder if we need to load a separate text encoder
        }

        print(f"[SeedVRModelLoader] Model loading complete.")
        return (seedvr_model_data, vae)


class SeedVRTextEmbedLoader:
    CATEGORY = "SeedVR"

    @classmethod
    def INPUT_TYPES(s):
        # Assuming embedding files are stored in ComfyUI's 'embeddings' folder
        # or a subfolder within it, or could be other checkpoint types.
        # For flexibility, let's allow selection from 'checkpoints' or 'embeddings'.
        # If specific .pt files are needed, users might need to place them in these folders.
        # Or, we can define a specific subfolder in our custom node for these.
        # For now, using 'embeddings' which is standard for CLIP embeds, etc.
        # If these are raw tensors not tied to CLIP, 'checkpoints' might be more appropriate.
        # Let's default to a new type 'seedvr_embeddings' for clarity.

        # Update: folder_paths does not have a generic 'seedvr_embeddings' type by default.
        # We should use a common existing type or allow any file path.
        # Using 'embeddings' for now, assuming .pt files can be placed there.

        # It's better to have a dedicated input for file names if they aren't discoverable
        # by folder_paths in a standard way.
        # For now, let's assume they are in 'embeddings' directory.

        # A more robust way might be to have string inputs for filenames and use a specific
        # subfolder within the custom node's directory, or within ComfyUI's input folder.
        # Let's use `folder_paths.get_filename_list("embeddings")` for now.

        embedding_files = folder_paths.get_filename_list("embeddings")
        if not embedding_files: # Provide a default if empty to avoid UI error
            embedding_files = ["example_embedding.pt"]

        return {
            "required": {
                "positive_embedding_name": (embedding_files,),
                "negative_embedding_name": (embedding_files,),
            }
        }

    RETURN_TYPES = ("EMBEDDINGS_POS", "EMBEDDINGS_NEG")
    FUNCTION = "load_embeddings"

    def load_embeddings(self, positive_embedding_name: str, negative_embedding_name: str):
        print(f"[SeedVRTextEmbedLoader] Loading text embeddings...")
        device = model_management.get_torch_device()

        # Get full paths to embedding files
        # These files are expected to be raw tensors or dicts containing tensors.
        pos_embed_path = folder_paths.get_full_path("embeddings", positive_embedding_name)
        neg_embed_path = folder_paths.get_full_path("embeddings", negative_embedding_name)

        if not pos_embed_path or not os.path.exists(pos_embed_path):
            raise FileNotFoundError(f"Positive embedding file not found: {positive_embedding_name} (resolved to {pos_embed_path})")
        if not neg_embed_path or not os.path.exists(neg_embed_path):
            raise FileNotFoundError(f"Negative embedding file not found: {negative_embedding_name} (resolved to {neg_embed_path})")

        print(f"[SeedVRTextEmbedLoader] Positive embedding: {pos_embed_path}")
        print(f"[SeedVRTextEmbedLoader] Negative embedding: {neg_embed_path}")

        try:
            # Load tensors
            # Assuming the .pt files directly contain the tensor or a dict with a known key like 'embed'
            pos_embed = torch.load(pos_embed_path, map_location="cpu")
            neg_embed = torch.load(neg_embed_path, map_location="cpu")

            # If they are dicts, extract the tensor
            if isinstance(pos_embed, dict):
                # Try common keys, or user needs to ensure the structure
                if "embed" in pos_embed: pos_embed = pos_embed["embed"]
                elif "embedding" in pos_embed: pos_embed = pos_embed["embedding"]
                else: raise ValueError(f"Positive embedding file {positive_embedding_name} is a dict but does not contain 'embed' or 'embedding' key.")

            if isinstance(neg_embed, dict):
                if "embed" in neg_embed: neg_embed = neg_embed["embed"]
                elif "embedding" in neg_embed: neg_embed = neg_embed["embedding"]
                else: raise ValueError(f"Negative embedding file {negative_embedding_name} is a dict but does not contain 'embed' or 'embedding' key.")

            if not isinstance(pos_embed, torch.Tensor) or not isinstance(neg_embed, torch.Tensor):
                raise TypeError("Embeddings must be PyTorch Tensors after loading.")

            # Move to device
            pos_embed = pos_embed.to(device)
            neg_embed = neg_embed.to(device)

            print(f"[SeedVRTextEmbedLoader] Embeddings loaded and moved to device: {device}")
            print(f"[SeedVRTextEmbedLoader] Pos shape: {pos_embed.shape}, Neg shape: {neg_embed.shape}")

        except Exception as e:
            print(f"[SeedVRTextEmbedLoader] Error loading embeddings: {e}")
            raise

        # The RETURN_TYPES "EMBEDDINGS_POS" and "EMBEDDINGS_NEG" are custom.
        # For ComfyUI, it's often best to return them as standard types like CONDITIONING
        # or specific tensor types if other nodes are designed to consume them.
        # For now, returning raw tensors. These might need to be wrapped or processed further
        # to be compatible with, e.g., CLIPSetEncode or KSampler inputs.
        # For this project, these embeddings are likely fed directly into the DiT model.
        return (pos_embed, neg_embed)

# NOTE: The following mappings should be in __init__.py, not here.
# NODE_CLASS_MAPPINGS = {
# "SeedVRModelLoader": SeedVRModelLoader,
# "SeedVRTextEmbedLoader": SeedVRTextEmbedLoader
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
# "SeedVRModelLoader": "SeedVR Model Loader",
# "SeedVRTextEmbedLoader": "SeedVR Text Embedding Loader"
# }
