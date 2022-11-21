# Anime-face-generator

![Fake images sample](https://github.com/PeterMatthew/Anime-face-generator/blob/main/fakes.jpg)

## How to use

Create venv and install requirements<br/>
`python -m venv .venv`<br/>
`source .venv/bin/activate` linux<br/>
`.venv\Scripts\Activate` windows<br/>
`pip install -r requirements.txt`

### Generate

`--config` path to config file<br/>
`--out` name of generated images<br/>
`--model` path to model<br/>

`python generate.py --config configs\stylegan_anime_64.yaml --out fakes.jpg --model models\gen64.pt`<br/>

### UI

inside ui folder enter `uvicorn main:app` then open `http://127.0.0.1:8000` in a browser<br/>

## References

Karras, T., Aila, T. and Laine, S. (2019) “A Style-Based Generator Architecture for
Generative Adversarial Networks”. Access: https://arxiv.org/pdf/1812.04948.pdf<br/>
Karras, T., Aila, T., Laine, S. and Lehtinen, J. (2018) “PROGRESSIVE GROWING OF
GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION”. Access: https://arxiv.org/pdf/1710.10196.pdf<br/>

### Dataset

dataset from https://www.kaggle.com/datasets/splcher/animefacedataset
