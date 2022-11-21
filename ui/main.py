from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from PIL import Image
import numpy as np
import io
import yaml
from stylegan import Generator
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../configs/stylegan_anime_64.yaml", "r") as data:
    config = yaml.safe_load(data)

model = Generator(64, 512, 512, device, config).to(device)
checkpoint = torch.load("../models/gen64.pt", map_location=device)
model.load_state_dict(checkpoint["g_ema_state_dict"])
w_mean = model.w_mean()

def gen_imgs():

    noise = torch.randn(1, 512, device=device)
    
    with torch.no_grad():
        fake = model(noise, 1, 64, False, truncation=0.6, truncation_w=w_mean)
    
    fake = F.tanh(fake)
    mean=0.5
    std=0.5
    
    img = np.transpose(fake[0].detach().cpu().numpy(), (1, 2, 0))
    img = (img*std + mean)*255
    img = img.astype(np.uint8)    

    return img

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request })


@app.get('/image/{id}/', response_class=Response)
async def get_image(id):
    
    arr = gen_imgs()
    im = Image.fromarray(arr)
    
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="test'+id+'.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')
