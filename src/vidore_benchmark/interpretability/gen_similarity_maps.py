import pprint
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image
from tqdm import trange

from vidore_benchmark.interpretability.colpali_processor import ColPaliProcessor
from vidore_benchmark.interpretability.plot_utils import plot_similarity_heatmap, plot_similarity_patches
from vidore_benchmark.interpretability.torch_utils import normalize_similarity_map_per_query_token
from vidore_benchmark.interpretability.vit_configs import VIT_CONFIG
from vidore_benchmark.models.colpali_model import ColPali
from vidore_benchmark.utils.constants import OUTPUT_DIR

SUPPORTED_PLOT_KINDS = ["patches", "heatmap"]


@dataclass
class InterpretabilityInput:
    query: str
    image: Image.Image
    start_idx_token: int
    end_idx_token: int


def gen_and_save_similarity_map_per_token(
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    image: Image.Image,
    kind: str = "heatmap",
    figsize: Tuple[int, int] = (8, 8),
    add_title: bool = True,
    style: str = "dark_background",
    savedir: str | Path | None = None,
) -> None:
    """
    Generate and save the similarity maps in the `outputs` directory for each token in the query.

    NOTE: The used device is the one specified in the model.
    """

    # Sanity checks
    if len(model.active_adapters()) != 1:
        raise ValueError("The model must have exactly one active adapter.")

    if model.config.name_or_path not in VIT_CONFIG:
        raise ValueError("The model must be referred to in the VIT_CONFIG dictionary.")
    vit_config = VIT_CONFIG[model.config.name_or_path]

    # Handle savepath
    if not savedir:
        savedir = OUTPUT_DIR / "interpretability" / str(uuid4())
        print(f"No savepath provided. Results will be saved to: `{savedir}`.")
    elif isinstance(savedir, str):
        savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # Resize the image to square
    input_image_square = image.resize((vit_config.resolution, vit_config.resolution))

    # Preprocess the inputs
    input_text_processed = processor.process_text(query, add_special_tokens=True).to(model.device)
    input_image_processed = processor.process_image(image, add_special_prompt=True).to(model.device)

    # Forward pass
    with torch.no_grad():
        output_text = model.forward(**asdict(input_text_processed))  # (1, n_text_tokens, hidden_dim)

    # NOTE: `output_image`` will have shape:
    # (1, n_patch_x * n_patch_y, hidden_dim) if `add_special_prompt` is False
    # (1, n_patch_x * n_patch_y + n_special_tokens, hidden_dim) if `add_special_prompt` is True
    with torch.no_grad():
        output_image = model.forward(**asdict(input_image_processed))

    # Remove the special tokens from the output
    output_image = output_image[:, : processor.processor.image_seq_length, :]  # (1, n_patch_x * n_patch_y, hidden_dim)

    output_image = rearrange(
        output_image, "b (h w) c -> b h w c", h=vit_config.n_patch_per_dim, w=vit_config.n_patch_per_dim
    )  # (1, n_patch_x, n_patch_y, hidden_dim)

    # Get the unnormalized attention map
    similarity_map = torch.einsum(
        "bnk,bijk->bnij", output_text, output_image
    )  # (1, n_text_tokens, n_patch_x, n_patch_y)
    similarity_map_normalized = normalize_similarity_map_per_query_token(
        similarity_map
    )  # (1, n_text_tokens, n_patch_x, n_patch_y)

    # Get text token information
    n_tokens = input_text_processed.input_ids.size(1)
    text_tokens = processor.tokenizer.tokenize(processor.decode(input_text_processed.input_ids[0]))
    print("\nText tokens:")
    pprint.pprint(text_tokens)
    print("\n")

    for token_idx in trange(1, n_tokens - 1, desc="Iterating over tokens..."):  # exclude the <bos> and the "\n" tokens
        if kind == "patches":
            fig, axis = plot_similarity_patches(
                input_image_square,
                vit_config.patch_size,
                vit_config.resolution,
                similarity_map=similarity_map_normalized[0, token_idx, :, :],
                figsize=figsize,
                style=style,
            )
            if add_title:
                fig.suptitle(f"Token #{token_idx}: `{text_tokens[token_idx]}`", color="white", fontsize=14)
        elif kind == "heatmap":
            fig, ax = plot_similarity_heatmap(
                input_image_square,
                vit_config.patch_size,
                vit_config.resolution,
                similarity_map=similarity_map_normalized[0, token_idx, :, :],
                figsize=figsize,
                style=style,
            )
            if add_title:
                ax.set_title(f"Token #{token_idx}: `{text_tokens[token_idx]}`", fontsize=14)
        else:
            raise ValueError(f"Invalid `kind` input: {kind}. Supported values are: {SUPPORTED_PLOT_KINDS}")

        savepath = savedir / f"token_{token_idx}.png"
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved attention map for token {token_idx} (`{text_tokens[token_idx]}`) to `{savepath}`.\n")
        plt.close(fig)

    return
