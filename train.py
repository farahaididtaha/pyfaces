import argparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    data = pickle.loads(open("models/encodings.pickle", "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["encodings"], labels)

    with open("models/recognizer.pickle", "wb") as f:
        f.write(pickle.dumps(recognizer))

    with open("models/le.pickle", "wb") as f:
        f.write(pickle.dumps(le))
