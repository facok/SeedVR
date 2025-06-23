from .nodes import SeedVRModelLoader, SeedVRTextEmbedLoader # Added SeedVRTextEmbedLoader

NODE_CLASS_MAPPINGS = {
    "SeedVRModelLoader": SeedVRModelLoader,
    "SeedVRTextEmbedLoader": SeedVRTextEmbedLoader # Added new node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVRModelLoader": "SeedVR Model Loader",
    "SeedVRTextEmbedLoader": "SeedVR Text Embed Loader" # Added display name
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
