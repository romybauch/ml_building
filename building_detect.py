
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import googlemaps
from google.cloud import vision, storage
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import tree, svm, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

key_map = "enter your google map key"
credentials = "enter your google cloud credentials"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
client_vision = vision.ImageAnnotatorClient()
client_maps = googlemaps.Client(key = key_map)
storage_client = storage.Client()


corr_label = \
    ["Construction", "Window", "Architecture",
     "Scaffolding", "Roof", "Classical architecture",
     "Medieval architecture", 'History', "Stately home",
     "Mansion", "Historic house", "Historic site",
     "Tower block", "Composite material", "Cement",
     "Sash window", "Landscape", "Human settlement",
     "Infrastructure", "Hacienda"]

my_labels = ["Construction" ,'Building', "Window", "Architecture", "Arch",
             "Urban design", "Plant", "Scaffolding", "Residential area",
          "Roof","Classical architecture", "Medieval architecture", 'History',
          "Stately home", "Mansion", "Balcony", "Historic house", "Fence",
          "Historic site","Condominium", "Tower block", "Composite material",
          "Electrical supply", "Cement", "Shed", "Arecales",
          "Landscape", "Asphalt",
          "Human settlement", "Metropolitan area", "Infrastructure", "Cottage",
              'name',"rate", "like", "Commercial building", "Tower block",
             "Flowerpot", "History", "Cobblestone", "Church", "Hacienda"]

chi_label = ["Construction", "Window", "Architecture",
             "Scaffolding", "Medieval architecture",
             "Fence",
          "Tower block", "Electrical supply", "Cement", "Shed", "Sash window",
             "Arecales", "Garden", "Landscape", "Ancient history", "Downtown",
             "Asphalt", "Human settlement", "Church"]

bucket_names = ["test-yuval-san", "presentation_route_1",
                "presentation_route_2"]

def detect_labels_uri(uri):
    """Detects labels in the file located in Google Cloud Storage or on the
    Web."""
    image = vision.Image()
    image.source.image_uri = uri

    response = client_vision.label_detection(image=image, max_results=50)

    if response.error.message:
        return None

    img_labels = response.label_annotations
    img_labels_string = []
    for i in range(len(img_labels)):
        img_labels_string.append(str(img_labels[i]))

    img_dict = {}

    for label in img_labels:
        # now its corr_label and not general labels
        if str(label.description) in corr_label:
            img_dict[str(label.description)] = label.score

    for label in corr_label:
        if label not in img_dict:
            img_dict[label] = 0

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return img_dict


def load_img():
    """Lists all the blobs in the bucket."""
    blobs = storage_client.list_blobs(bucket_names[0])

    row_list = []

    for blob in blobs:
        uri = "gs://" + bucket_names[0] + "/" + str(blob.name)
        img_dic_temp = detect_labels_uri(uri)

        img_dic_temp['name'] = str(blob.name)
        row_list.append(img_dic_temp)

    df = pd.DataFrame(row_list)
    df['like'] = ""
    dir_to_send_data = "enter your dir"
    df.to_csv(dir_to_send_data)


def compare_routes(classification_model):
    bucket_route1 = bucket_names[1]
    bucket_route2 = bucket_names[2]

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs1 = storage_client.list_blobs(bucket_route1)
    blobs2 = storage_client.list_blobs(bucket_route2)

    route_1 = pd.DataFrame(columns=corr_label)
    route_2 = pd.DataFrame(columns=corr_label)
    for blob in blobs1:
        uri = "gs://" + bucket_route1 + "/" + str(blob.name)
        img_dic_temp = detect_labels_uri(uri)
        route_1 = route_1.append(img_dic_temp, ignore_index=True)

    print(classification_model.predict(route_1))
    counter1 = classification_model.predict(route_1).sum()

    for blob in blobs2:
        uri = "gs://" + bucket_route2 + "/" + str(blob.name)
        img_dic_temp = detect_labels_uri(uri)
        route_2 = route_2.append(img_dic_temp, ignore_index=True)

    print(classification_model.predict(route_2))
    counter2 = classification_model.predict(route_2).sum()

    if counter1>counter2:
        print("route 1 is prettier")
    else:
        print("route 2 is prettier")
    print("route 1 rate is: " + str(counter1))
    print("route 2 rate is: " + str(counter2))


def classification_predict(X_train, X_test, Y_train, Y_test):
    tree_predict(X_train, X_test, Y_train, Y_test)
    knn_predict(X_train, X_test, Y_train, Y_test)
    svm_predict(X_train, X_test, Y_train, Y_test)


def tree_predict(X_train, X_test, Y_train, Y_test):
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    y_pred = dtc.predict(X_test)
    pr = metrics.accuracy_score(Y_test, y_pred)
    print(f"Tree model: {pr:.2f}% accuracy")


def knn_predict(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    pr = metrics.accuracy_score(Y_test, y_pred)
    print(f"7-NN model: {pr:.2f}% accuracy")


def svm_predict(X_train, X_test, Y_train, Y_test):
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, Y_train)
    y_pred = svm_clf.predict(X_test)
    pr = metrics.accuracy_score(Y_test, y_pred)
    print(f"SVM model: {pr:.2f}% accuracy")


def feature_correlation(X,y):
    feature_dict = {}
    X.insert(0, 'like', y)
    cov_XY = X.cov()['like']
    std_y = y.std()
    X.sort_values(by=['like'])

    for col in X.columns:
        corr = float(cov_XY[col] / (std_y * (X[col].std())))
        feature_dict[col] = corr

    print(sorted(((v, k) for k, v in feature_dict.items()), reverse=True))
    del X['like']


def feature_chi2(x, y):
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(x,y)
    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(x)
    # Summarize selected features
    print(features)
    print(features[0:10, :])


def feature_correlation_each_other(df):
    # plot heatmap
    plt.figure(figsize=(30, 21))

    # Generate a mask to onlyshow the bottom triangle
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # generate heatmap
    sns.heatmap(df.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Coefficient Of Predictors')
    plt.show()


def load_route():
    # change place_id to start and end point of your routh
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin=" \
          f"place_id:ChIJy-5EkScoAxURgxq8bnaOMQo&destination=place_id:ChIJW" \
          f"dChltTXAhURwKcI-93tcgo&mode=walking&alternatives=true&key={key_map}"
    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload).json()
    json.dumps(response)
    with open("routes.json",'w') as fp:
        json.dump(response,fp)


def detect_address_in_routes():
    data=None
    with open('routes.json') as json_file:
        data = json.load(json_file)

    steps = []
    for rt in data['routes']:
        s = []
        for step in rt['legs'][0]['steps']:
            lat = step['start_location']['lat']
            lng = step['start_location']['lng']
            s.append((lat, lng))
        steps.append(s)
    print(steps)


if __name__ == '__main__':

    load_img()
    import_dir_data = "enter your dir"
    df = pd.DataFrame(pd.read_csv(import_dir_data))
    df.fillna(0)
    target = df["like"]

    # evaluate features:
    feature_correlation(df, target)
    feature_correlation_each_other(df)
    feature_chi2(df,target)

    data = df[corr_label]

    # split data
    x_train, x_test, y_train, y_test = \
        train_test_split(data, target, test_size=0.25)

    # find best classification
    classification_predict(x_train, x_test, y_train, y_test)
    svm_clf = svm.SVC()
    svm_clf.fit(data, target)
    compare_routes(svm_clf)

    # find route by google maps and extract addresses
    load_route()
    detect_address_in_routes()
