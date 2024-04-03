import os
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# directory where the images are located
plan_path = r"FloorPlans"
# directory to save figures
fig_path = "Figures"
# directory to download save models
model_path = "Models"
# this list holds all the image filenames
images = []

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


def extract_features(file, model):
    img = load_img(plan_path + "/" + file, target_size=(224, 224))
    # convert to numpy array and reshape for model
    reshaped_img = np.array(img).reshape(1, 224, 224, 3)
    # prepare image for model
    processed_img = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(processed_img)
    return features


data = {}
# p = r".\FeatureVectors"

# loop through each image in the dataset
for image in images:
    # try to extract the features and update the dictionary
    features = extract_features(image, model)
    data[image] = features


# get a list of the filenames
filenames = np.array(list(data.keys()))

# get and reshape a list of just the features
features = np.array(list(data.values())).reshape(-1, 4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(features)
x = pca.transform(features)
dump(pca, f"{model_path}/pca.joblib")

# cluster feature vectors
kmeans = KMeans(n_clusters=3, random_state=22)
kmeans.fit(x)
dump(kmeans, f"{model_path}/kmeans.joblib")

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
