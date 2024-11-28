# Paper reproduction instructions

## Instructions

If you are willing to reproduce the results from the ColPali paper, you should:

- clone the repository
- checkout to the `3.3.0` tag or below
- create a new virtual environment
- copy the list of package versions from this file to a `requirements-dev.txt` file
- install the dependencies using the following command:

```bash
pip install -r requirements-dev.txt
```

Then, you can run the CLI commands to evaluate the models and reproduce the results from the paper.

## Package versions

`requirements-dev.txt`:

```
accelerate==0.30.1
aiofiles==23.2.1
aiohttp==3.9.5
aiosignal==1.3.1
altair==5.3.0
annotated-types==0.7.0
anyio==4.4.0
asttokens==2.4.1
attrs==23.2.0
black==24.4.2
certifi==2024.6.2
charset-normalizer==3.3.2
click==8.1.7
comm==0.2.2
configue==5.0.0
contourpy==1.2.1
cycler==0.12.1
datasets==2.19.1
debugpy==1.8.1
decorator==5.1.1
dill==0.3.8
diskcache==5.6.3
dnspython==2.6.1
einops==0.8.0
email_validator==2.2.0
eval_type_backport==0.2.0
executing==2.0.1
fastapi==0.111.0
fastapi-cli==0.0.4
ffmpy==0.3.2
filelock==3.15.3
FlagEmbedding==1.2.10
fonttools==4.53.0
frozenlist==1.4.1
fsspec==2024.3.1
gradio==4.36.1
gradio_client==1.0.1
h11==0.14.0
httpcore==1.0.5
httptools==0.6.1
httpx==0.27.0
huggingface-hub==0.23.4
idna==3.7
importlib_resources==6.4.0
iniconfig==2.0.0
ipykernel==6.29.4
ipython==8.25.0
jedi==0.19.1
Jinja2==3.1.4
joblib==1.4.2
jsonlines==4.0.0
jsonschema==4.22.0
jsonschema-specifications==2023.12.1
jupyter_client==8.6.2
jupyter_core==5.7.2
kiwisolver==1.4.5
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.0
matplotlib-inline==0.1.7
mdurl==0.1.2
mpmath==1.3.0
mteb==1.12.47
multidict==6.0.5
multiprocess==0.70.16
mypy-extensions==1.0.0
nest-asyncio==1.6.0
networkx==3.3
nltk==3.8.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.5.40
nvidia-nvtx-cu12==12.1.105
orjson==3.10.5
packaging==24.1
pandas==2.2.2
parso==0.8.4
pathspec==0.12.1
pdf2image==1.17.0
peft==0.11.1
pexpect==4.9.0
pillow==10.3.0
platformdirs==4.2.2
pluggy==1.5.0
polars==0.20.31
prompt_toolkit==3.0.47
protobuf==5.27.1
psutil==6.0.0
ptyprocess==0.7.0
pure-eval==0.2.2
pyarrow==16.1.0
pyarrow-hotfix==0.6
pydantic==2.7.4
pydantic_core==2.18.4
pydub==0.25.1
Pygments==2.18.0
pyparsing==3.1.2
pytesseract==0.3.10
pytest==8.2.2
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-multipart==0.0.9
pytrec-eval-terrier==0.5.6
pytz==2024.1
PyYAML==6.0.1
pyzmq==26.0.3
rank-bm25==0.2.2
referencing==0.35.1
regex==2024.5.15
requests==2.32.3
rich==13.7.1
rpds-py==0.18.1
ruff==0.4.10
safetensors==0.4.3
scikit-learn==1.5.0
scipy==1.13.1
seaborn==0.13.2
semantic-version==2.10.0
sentence-transformers==3.0.1
sentencepiece==0.2.0
shellingham==1.5.4
six==1.16.0
sniffio==1.3.1
stack-data==0.6.3
starlette==0.37.2
sympy==1.12.1
threadpoolctl==3.5.0
timm==1.0.7
tokenizers==0.19.1
tomlkit==0.12.0
toolz==0.12.1
torch==2.3.1
torchvision==0.18.1
tornado==6.4.1
tqdm==4.66.4
traitlets==5.14.3
transformers==4.41.2
triton==2.3.1
typer==0.12.3
typing_extensions==4.12.2
tzdata==2024.1
ujson==5.10.0
urllib3==2.2.2
uvicorn==0.30.1
uvloop==0.19.0
vidore-benchmark==1.0.0
watchfiles==0.22.0
wcwidth==0.2.13
websockets==11.0.3
xxhash==3.4.1
yarl==1.9.4
```
