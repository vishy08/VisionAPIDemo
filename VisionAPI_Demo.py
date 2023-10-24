import os, io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'

client = vision_v1.ImageAnnotatorClient()

def detectText(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content = content)

    response = client.text_detection(image=image)

    texts = response.text_annotations

    df = pd.DataFrame(columns=['locale', 'description'])

    for text in texts:
        df = df.append(
            dict(
                Locale = text.locale,
                description = text.description
            ),
            ignore_index=True
        )

    return df['description'][0]

def printText(df):
    print(df)
    return df



FILE_NAME = 'vishyOneCard.jpg'
FOLDER_PATH = r'/Users/vishalkasula/Desktop/PythonVenv/VisionAPIDemo/images/text/'
printText(detectText(os.path.join(FOLDER_PATH, FILE_NAME)))