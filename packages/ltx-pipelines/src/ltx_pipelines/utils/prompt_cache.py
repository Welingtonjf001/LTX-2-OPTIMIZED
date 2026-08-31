import hashlib
import os


DEFAULT_PROMPT_CACHE_DIR = "./prompt_embeddings_cache"


def prompt_cache_path(
    prompt: str,
    *,
    enhance_prompt: bool = False,
    seed: int = 0,
    image_identifier: str = "no_img",
    cache_dir: str = DEFAULT_PROMPT_CACHE_DIR,
) -> str:
    hash_input = (
        f"prompt:{prompt}|"
        "pipeline:music_distilled|"
        f"enhance:{enhance_prompt}|"
        f"seed:{seed if enhance_prompt else 'ignored'}|"
        f"img:{image_identifier}"
    )
    cache_filename = hashlib.md5(hash_input.encode("utf-8")).hexdigest() + ".pt"
    return os.path.join(cache_dir, cache_filename)
