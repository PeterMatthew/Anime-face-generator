# Anime-face-generator

![Fake images sample](https://github.com/PeterMatthew/Anime-face-generator/blob/main/fakes.jpg)

## How to use

### Generate

`--config` path to config file<br/>
`--out` name of generated images<br/>
`--model` path to model<br/>

`python generate.py --config configs\stylegan_anime_64.yaml --out fakes.jpg --model models\gen64.pt`<br/>

### UI

inside ui folder enter `uvicorn main:app` then open `http://127.0.0.1:8000` in a browser<br/>
