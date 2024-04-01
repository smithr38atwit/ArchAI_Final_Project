import io
import os
import pickle
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cairosvg import svg2png
from joblib import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

path = r"FloorPlans"
# change the working directory to the path where the images are located
# os.chdir(path)

# this list holds all the image filename
images = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith(".png"):
            # adds only the image files to the flowers list
            images.append(file.name)


model = VGG16()
model.save("Models/vgg16.keras")
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.save("Models/vgg16_trunc.keras")

# def svgRead(filename):
#     """Load an SVG file and return image in Numpy array"""
#     # Make memory buffer
#     mem = io.BytesIO()
#     # Convert SVG to PNG in memory
#     svg2png(url=filename, write_to=mem)
#     # Convert PNG to Numpy array
#     return np.array(Image.open(mem))


def extract_features(file, model):
    # img = svgRead(file)
    img = load_img(path + "\\" + file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
# p = r".\FeatureVectors"

# lop through each image in the dataset
for image in images:
    # try to extract the features and update the dictionary
    feat = extract_features(image, model)
    data[image] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    # except:
    #     with open(p, "wb") as file:
    #         pickle.dump(data, file)


# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 119 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)
dump(pca, "Models/pca.joblib")

# cluster feature vectors
kmeans = KMeans(n_clusters=3, random_state=22)
kmeans.fit(x)
dump(kmeans, "Models/kmeans.joblib")

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 3:
        print(f"Clipping cluster size from {len(files)} to 3")
        files = files[:3]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(1, 3, index + 1)
        img = load_img(path + "\\" + file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"Figures/Cluster{cluster}_small")


for key in groups.keys():
    view_cluster(key)


# this is just incase you want to see which value for k might be the best
# sse = []
# list_k = list(range(3, 50))

# for k in list_k:
#     km = KMeans(n_clusters=k, random_state=22)
#     km.fit(x)

#     sse.append(km.inertia_)

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r"Number of clusters *k*")
# plt.ylabel("Sum of squared distance")
