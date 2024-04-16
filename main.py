import io
import os

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

# directory where the images are located
plan_path = "cubicasa5k"
# directory to save figures
fig_path = "Figures"
# directory to download save models
model_path = "Models"
# this list holds all the image filenames
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


# Extract features from each image
print("Extracting features...")
data = {}
for i, image in enumerate(images):
    print(f"{i+1} / {len(images)}")
    features = extract_features(image, model)
    if features is None:
        continue
    data[image] = features

filenames = np.array(list(data.keys()))
features = np.array(list(data.values())).reshape(-1, 4096)

# Reduce the amount of dimensions in the feature vector
print("Performing PCA...")
pca = PCA(n_components=100, random_state=22)
pca.fit(features)
x = pca.transform(features)
dump(pca, f"{model_path}/pca.joblib")

# Cluster feature vectors
print("Performing K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=22)
kmeans.fit(x)
dump(kmeans, f"{model_path}/kmeans.joblib")


print("Visualizing clusters...")

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    groups[cluster].append(file)


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]

    if not os.path.exists(f"{fig_path}"):
        os.makedirs(f"{fig_path}")
    # only allow up to 30 images to be shown at a time
    if len(files) > 3:
        print(f"Clipping cluster size from {len(files)} to 3")
        files = files[:3]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(1, 3, index + 1)
        if plan_path == "cubicasa5k":
            img = svgRead(file)
        else:
            img = load_img(plan_path + "/" + file)
            img = np.array(img)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"{fig_path}/Cluster{cluster}_small")


for key in groups.keys():
    view_cluster(key)


# determine value for k
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
