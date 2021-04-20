## Project Structure
- Face Detection
- Gender Detection
- Emotion Detection

## How to install
```angular2html
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
## How to run
- Face Detection

```angular2html
python detect_image.py --image data/cristiano.jpg --method cnn
```
> `--methods` can be one of `haar`, `dnn`, `hog`, `cnn`
- Gender Detection

- Emotion Detection

- How to generate the database

- Face Recognition

## TODO
- Add PreCommit
- Add Typing Hint
- Add Custom Error or Exception Class
- Add Bounding Classes for Face Detection
