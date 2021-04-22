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

## TODO
- Add PreCommit
- Add Gitignore
- Add Typing Hint
- Add Custom Error or Exception Class
- Add Bounding Classes for Face Detection
