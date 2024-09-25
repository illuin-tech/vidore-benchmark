from pathlib import Path
from typing import Annotated, List, cast

import torch
import typer
from colpali_engine.models import ColPali, ColPaliProcessor
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from vidore_benchmark.interpretability.gen_similarity_maps import gen_and_save_similarity_map_per_token
from vidore_benchmark.utils.constants import OUTPUT_DIR
from vidore_benchmark.utils.logging_utils import setup_logging
from vidore_benchmark.utils.torch_utils import get_torch_device

load_dotenv(override=True)


app = typer.Typer(
    help="CLI for generating similarity maps for ColPali.",
    no_args_is_help=True,
)


@app.command()
def generate_similarity_maps(
    documents: Annotated[List[Path], typer.Option(help="List of document filepaths (image format)")],
    queries: Annotated[List[str], typer.Option(help="List of queries")],
    device: Annotated[str, typer.Option(help="Torch device")] = "auto",
    log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning",
) -> None:
    """
    Load the ColPali model and, for each query-document pair, generate similarity maps fo
    each token in the current query.
    """

    logger.enable("vidore_benchmark")
    setup_logging(log_level)

    # Sanity checks
    assert len(documents) == len(queries), "The number of documents and queries must be the same."
    for document in documents:
        if not document.is_file():
            raise FileNotFoundError(f"File not found: `{document}`")

    # Set the device
    device = get_torch_device(device)
    print(f"Using device: {device}")

    # Load the model and LORA adapter
    model_name = "vidore/colpali-v1.2"
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    )

    # Load the processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    print("Loaded custom processor.\n")

    images = [Image.open(img_filepath) for img_filepath in documents]

    for query, image, filepath in zip(queries, images, documents):
        print(f"\n\nProcessing query `{query}` and document `{filepath}`\n")
        savedir = OUTPUT_DIR / "interpretability" / filepath.stem
        gen_and_save_similarity_map_per_token(
            model=model,
            processor=processor,
            query=query,
            image=image,
            savedir=savedir,
        )

    print("\nDone.")


if __name__ == "__main__":
    app()
