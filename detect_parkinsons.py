#Import All The Required Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def quantify_image(image):
    #We Calculate the HOG of the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    
    #Return Feature Vector
    return features

def load_split(path):
    #Grab List Of Images In The Input Directory
    #Initialise List Of Data And Class Labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    #Loop Over Image Paths
    for imagePath in imagePaths:
        #Extract Class Label From Class Name
        label = imagePath.split(os.path.sep)[-2]

        #Load Input Image, Convert To Greyscale, Resize To 
        #200x200px
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        #Threshold The Image
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #Quantify Image
        features = quantify_image(image)

        #Update Data And Library Lists
        data.append(features)
        labels.append(label)

    #Return Data And Labels
    return(np.array(data), np.array(labels))

#Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-t", "--trials", type = int, default=5,
    help="# of trials to run")
args = vars(ap.parse_args())

#Define Path To Training And Testing Directories
trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

#Load Training And Testing Data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

#Encode Labels As Interface
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

#Initialize Our Trials Directory
trials = {}

#Loop Over Number Of Trials To Run
for i in range(0, args["trials"]):
    #Train Model
    print("[INFO] training model {} of {}...".format(i+1,
        args["trials"]))
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)

#Make Predictions On Testing Data, Initialize Dictionary
#To Store Computations
predictions = model.predict(testX)
metrics = {}

#Compute Confusion Matrix
#Derieve Raw Accuracy, Sensitivyt, And Specificity
cm = confusion_matrix(testY, predictions).flatten()
(tn, fp, fn, tp) = cm
metrics["acc"] = (tp+tn) / float(cm.sum())
metrics["sensitivity"] = tp / float(tp+fn)
metrics["specificity"] = tn / float(tn+fn)

#Loop Over Metrics
for (k, v) in metrics.items():
    #Update Trials Directionary With List Of Values For
    #Current Metrics
    l = trials.get(kk, [])
    l.append(v)
    trials[k] = l

#Loop Over Matrix
for metric in ("acc", "sensitivity", "specificity"):
    #Take List Of Values For Matrics,
    #Calculate Mean And Std Deviation
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)

    #Output The Computation
    print(metric)
    print("="* len(metric))
    print("u={:.4f}, o={:.4f}".format(mean, std))
    print("")

#Randomly Select Few Images, Initialize Output Images
#For Maontage
testingPaths = list(paths.list_images(testingPath))
idxs = np.arrange(0, len(testingPath))
idxs = np.random.choice(idxs, size=(25,), replace=False)
images=[]

#Loop Over Testing Samples
for i in idxs:
    #Load testing image, clone it, and resize it
    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))

    #Pre-process Image
    image = cv2.cvtColor(image, COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #Quantify Image, Make Predictions On Extracted Features
    features = quantify_image(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]

    #Draw Colored Class Label On Output Image
    #Add It To Output Images
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5,
        color, 2)
    images.append(output)

#Create Montage(128x128) with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]

#Show Output Montage
cv2.imshow("Output", montage)
cv2.waitKey(0)