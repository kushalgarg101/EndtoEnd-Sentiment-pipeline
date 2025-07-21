from fastapi import FastAPI
from inference import Infer
from typing import List, Dict, Tuple, Annotated, Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post('/predictions')
def get_preds(input_data : List[str]):
    init_pred = Infer(input_data)
    labels = init_pred.model_outputs()
    return {"predictions": labels}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()