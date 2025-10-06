from fastapi import FastAPI, File ,UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

MODEL = tf.keras.models.load_model("models/potato_disease_model_1.keras")
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']

app = FastAPI()
@app.get("/ping")

async def ping():
    return "Hello i am ping"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file : UploadFile = File(...)
):
   
   image = read_file_as_image(await file.read())
  
  # the image is 1d arrray we need to convert it to 2d by function expand dims
   image_batch = np.expand_dims(image, 0)

   predictions = MODEL.predict(image_batch)   
   predicted_class_index = np.argmax(predictions[0]) 
   predicted_class = CLASS_NAMES[predicted_class_index]
   confidence = np.max(predictions[0])
   return {
        'class': predicted_class,
        # Convert NumPy float to standard Python float for proper JSON serialization
        'confidence': float(confidence) 
    }


if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port = 8000)
    
