## Project Structure
- Face Detection
- Gender Detection
- Emotion Detection

## How to install
```angular2html
git clone https://github.com/farahaididtaha/pyfaces.git
cd pyfaces
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
## How to run
- How to gather image dataset
```angular2html
pwd
export PATH=$PATH:<directory>/preprocessing/drivers
python gather_images.py -q "messi" -n 10
python gather_images.py -q "neymar" -n 10
```
> The images will be downloaded into `data` directory. Currently this script is not fully-fledged so once you scrape, please double check.

- Face Detection

```angular2html
python detect_image.py --image data/cristiano/00.jpg --method dnn
```
> `--methods` can be one of `haar`, `dnn`, `hog`, `cnn`
- Gender Detection

- Emotion Detection

- How to generate the database

- Face Recognition
1. We need to extract the faces from the collected dataset
```angular2html
python extract_faces.py
```
> This will extract the faces from our dataset and save the faces into another directory. Result will be written into `data/output` directory.

```angular2html
python encode_faces.py
```
> This extract the embeddings(128-d vector) from the face images.

```angular2html
python train.py
```
> This will train the model using calculated embeddings

```angular2html
python main.py --image test/test.jpg
python main.py --image test/test2.jpg
```
> This will train the model using calculated embeddings

```angular2html
python main.py --video test/video.jpg
python main.py --image test/test2.jpg
```
> This will train the model using calculated embeddings


## TODO
- Add PreCommit
- Add Configuration File
- Add Typing Hint
- Add Custom Error or Exception Class
- Add Bounding Classes for Face Detection
- Add My own Gender Detection Model
- Add My own Emotional Detection
- Add Pipeline for generation
- Update Preprocessing Packages
    - Build more scrapping approaches
    - Find out duplicate images
- Add RetinaFace for face detection
- Add ArcFace for face recognition
- Add metrics function to show the several detection & recognition algorithms
  