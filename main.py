import io
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from cairosvg import svg2png
from joblib import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plan_path = "cubicasa5k"  # Floor plans path
fig_path = "Figures"  # Saved figures path
model_path = "Models"  # Saved models path
load_features = True  # Whether to load image features or extract them
n_components = 100  # Number of components for PCA
n_clusters = 30  # Number of clusters for K-Means

images = []

if plan_path == "cubicasa5k":
    with open(os.path.join(plan_path, "all.txt")) as file:
        images = [f"{plan_path}{line.rstrip()}model.svg" for line in file]
else:
    # creates a ScandirIterator aliased as files
    with os.scandir(plan_path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith(".jpg"):
                # adds only the image files to the flowers list
                images.append(file.name)


if not os.path.exists(f"{model_path}"):
    os.makedirs(f"{model_path}")

model = VGG16()
model.save(f"{model_path}/vgg16.keras")
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.save(f"{model_path}/vgg16_trunc.keras")


def svgRead(filename):
    """Load an SVG file and return image in Numpy array"""
    # Make memory buffer
    mem = io.BytesIO()
    # Convert SVG to PNG in memory
    try:
        svg2png(url=filename, write_to=mem, output_height=224, output_width=224)
    except:
        with open("cubicasa5k/bad.txt", "a") as bad:
            bad.write(filename + "\n")
        return None
    # Convert PNG to Numpy array
    return np.array(Image.open(mem))[:, :, :-1]


def extract_features(file, model):
    if plan_path == "cubicasa5k":
        img = svgRead(file)
        if img is None:
            return None
        reshaped_img = img.reshape(1, 224, 224, 3)
    else:
        img = load_img(plan_path + "/" + file, target_size=(224, 224))
        # convert to numpy array and reshape for model
        reshaped_img = np.array(img).reshape(1, 224, 224, 3)
    # prepare image for model
    processed_img = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(processed_img, verbose=0)
    return features


data = {}
if load_features:
    with open(f"{model_path}/extracted_features.pickle", "rb") as f:
        data = pickle.load(f)
else:
    # Extract features from each image
    log_interval = int(len(images) * 0.1)
    for i, image in enumerate(images):
        if i + 1 % log_interval == 0:
            print(f"{i+1} / {len(images)}")
        features = extract_features(image, model)
        if features is None:
            continue
        data[image] = features

    with open(f"{model_path}/extracted_features.pickle", "wb") as f:
        pickle.dump(data, f)

filenames = np.array(list(data.keys()))
features = np.array(list(data.values())).reshape(-1, 4096)

# Reduce the amount of dimensions in the feature vector
print("Performing PCA...")
pca = PCA(n_components=n_components, random_state=42)
pca.fit(features)
x = pca.transform(features)
dump(pca, f"{model_path}/pca.joblib")

# Cluster feature vectors
print("Performing K-Means Clustering...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(x)
dump(kmeans, f"{model_path}/kmeans.joblib")


print("Visualizing clusters...")

# Create directories
if not os.path.exists(f"{fig_path}"):
    os.makedirs(f"{fig_path}")
identifier = f"{n_components}co_{n_clusters}cl"
if not os.path.exists(f"{fig_path}/{identifier}"):
    os.makedirs(f"{fig_path}/{identifier}")

# holds the cluster id and the images { id: {num_files: n, files: [images]} }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    cluster = int(cluster)
    if cluster not in groups.keys():
        groups[cluster] = {"num_files": 0, "files": []}
    groups[cluster]["num_files"] += 1
    groups[cluster]["files"].append(file)

json_path = f"{fig_path}/{identifier}.json"
if os.path.exists(json_path):
    os.remove(json_path)
with open(json_path, "w") as f:
    json.dump(groups, f, sort_keys=True, indent=4)


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]["files"]

    if not os.path.exists(f"{fig_path}"):
        os.makedirs(f"{fig_path}")
    # only allow up to n_files images to be shown at a time
    n_files = 9
    if len(files) > n_files:
        print(f"Clipping cluster size from {len(files)} to {n_files}")
        files = files[:n_files]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(3, 3, index + 1)
        if plan_path == "cubicasa5k":
            img = svgRead(file)
        else:
            img = load_img(plan_path + "/" + file)
            img = np.array(img)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"{fig_path}/{identifier}/Cluster{cluster}_small")
    plt.close()


for key in groups.keys():
    view_cluster(key)
