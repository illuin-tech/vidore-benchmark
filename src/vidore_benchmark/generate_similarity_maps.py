from pathlib import Path
from typing import Annotated, List, cast

import typer
from dotenv import load_dotenv
from PIL import Image

from vidore_benchmark.interpretability.colpali_processor import ColPaliProcessor
from vidore_benchmark.interpretability.gen_similarity_maps import gen_and_save_similarity_map_per_token
from vidore_benchmark.models.colpali_model import ColPali
from vidore_benchmark.utils.constants import OUTPUT_DIR
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
) -> None:
    """
    Load the ColPali model and, for each query-document pair, generate similarity maps fo
    each token in the current query.
    """

    # Sanity checks
    assert len(documents) == len(queries), "The number of documents and queries must be the same."
    for document in documents:
        assert document.is_file(), f"File not found: `{document}`"

    device = get_torch_device(device)

    model_path = "google/paligemma-3b-mix-448"
    lora_path = "vidore/colpali"

    # Load the model and LORA adapter
    model = cast(ColPali, ColPali.from_pretrained(model_path, device_map=device))

    # Load the Lora adapter into the model
    # Note:`add_adapter` is used to create a new adapter while `load_adapter` is used to load an existing adapter
    model.load_adapter(lora_path, adapter_name="colpali", device_map=device)
    if model.active_adapters() != ["colpali"]:
        raise ValueError(f"Incorrect adapters loaded: {model.active_adapters()}")
    print(f"Loaded model from {model_path} and LORA from {lora_path}.")

    # Load the processor
    processor = ColPaliProcessor.from_pretrained(model_path)
    print("Loaded custom processor.\n")

    assert all([img_filepath.is_file() for img_filepath in documents])

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
