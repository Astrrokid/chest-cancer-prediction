import os
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory='templates')

class ClientApp:
    def __init__(self) -> None:
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()
@app.get('/', response_class=HTMLResponse)
def home():
    return templates.TemplateResponse('index.html', {'request': {}})


@app.post('/train')
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"

@app.post('/predict')
async def predictRoute(request: Request):
    payload = await request.json()
    image = payload.get("image")
    decodeImage(image, filename=clApp.filename)
    result = clApp.classifier.predict()
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)