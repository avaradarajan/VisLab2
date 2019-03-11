import json

from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
#First of all you have to import it from the flask module:
app = Flask(__name__)
#By default, a route only answers to GET requests. You can use the methods argument of the route() decorator to handle different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    global df
    print(df.shape)
    #TASK 1 - Part 1 - Random Sampling
    random_sample = df.sample(n=500)
    #print(random_sample)
    records = random_sample.to_dict(orient='records')
    #print(records)
    jsonRecords = json.dumps(records, indent=2)
    #print(jsonRecords)

    #Task 1 - Part 2 - Stratified Sampling
    #Normalization

    '''before = preprocessing.MinMaxScaler()
    after = before.fit(df)
    normalized_df = pd.DataFrame(after.transform(df))
    print(normalized_df)'''
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df))
    #after = preprocessing.normalize(df)
    #normalized_df = pd.DataFrame(after)
    print(normalized_df)

    km = []
    #Training K
    for k in range(1,15):
        kmeans = KMeans(n_clusters=k,random_state=1)
        kmeans.fit(normalized_df)
        km.append((k,kmeans.inertia_/1000000))
    data = pd.DataFrame(km, columns=list("ki"))
    #Reference - https://stackoverflow.com/questions/51762514/find-the-elbow-point-on-an-optimization-curve-with-python
    kn = KneeLocator(data.k, data.i, curve='convex', direction='decreasing')
    print(kn.knee)
    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata}
    return render_template("index.html",data=data)


@app.route("/strat", methods=['POST', 'GET'])
def strat():

    global df

    if request.method == 'POST':
        if request.form['data'] == 'before':
            scaler = StandardScaler()
            normalized_df = pd.DataFrame(scaler.fit_transform(df))
            # PCA - Task 2
            pca = PCA(n_components=20)
            components = pca.fit_transform(normalized_df)
            print((pca.explained_variance_ratio_ * 100))
            variance = []
            sum = 0
            for index, val in enumerate(pca.explained_variance_):
                variance.append((index + 1, val))
                # sum += pca.explained_variance_ratio_[index]
            # print(sum)
            pdf = pd.DataFrame(data=components,
                               columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10',
                                        'PCA11', 'PCA12', 'PCA13', 'PCA14' \
                                   , 'PCA15', 'PCA16', 'PCA17', 'PCA18', 'PCA19', 'PCA20'])
            # print(pdf)
            data = pd.DataFrame(variance, columns=list("ki"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data) # Should be a json string
        elif request.form['data'] == 'after':
            # After training
            scaler = StandardScaler()
            normalized_df = pd.DataFrame(scaler.fit_transform(df))
            #print(normalized_df)
            kmeans = KMeans(n_clusters=5, random_state=1)
            kmeans.fit(normalized_df)
            #print(kmeans.labels_)
            normalized_df['clusters'] = kmeans.labels_
            cluster1Data = normalized_df.loc[normalized_df['clusters'] == 0];
            cluster2Data = normalized_df.loc[normalized_df['clusters'] == 1];
            cluster3Data = normalized_df.loc[normalized_df['clusters'] == 2];
            cluster4Data = normalized_df.loc[normalized_df['clusters'] == 3];
            cluster5Data = normalized_df.loc[normalized_df['clusters'] == 4];
            #print(type(cluster2Data))
            sampleFromCluster1 = cluster1Data.sample(frac=0.50)
            sampleFromCluster2 = cluster2Data.sample(frac=0.50)
            sampleFromCluster3 = cluster3Data.sample(frac=0.50)
            sampleFromCluster4 = cluster4Data.sample(frac=0.50)
            sampleFromCluster5 = cluster5Data.sample(frac=0.50)
            finalSample = sampleFromCluster1.append(sampleFromCluster2).append(sampleFromCluster3).append(sampleFromCluster4).append(sampleFromCluster5);
            finalSample = finalSample.drop(columns="clusters")

            #PCA - Task 2
            pca = PCA(n_components=20)
            components = pca.fit_transform(finalSample)
            print((pca.explained_variance_ratio_*100))
            variance = []
            sum=0
            for index,val in enumerate(pca.explained_variance_):
                variance.append((index+1,val))
                #sum += pca.explained_variance_ratio_[index]
            #print(sum)
            pdf = pd.DataFrame(data=components,columns=['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10','PCA11','PCA12','PCA13','PCA14' \
                ,'PCA15','PCA16','PCA17','PCA18','PCA19','PCA20'])
            #print(pdf)
            data = pd.DataFrame(variance, columns=list("ki"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data)

    # After training
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df))
    # print(normalized_df)
    kmeans = KMeans(n_clusters=5, random_state=1)
    kmeans.fit(df)
    # print(kmeans.labels_)
    df['clusters'] = kmeans.labels_
    cluster1Data = df.loc[df['clusters'] == 0];
    cluster2Data = df.loc[df['clusters'] == 1];
    cluster3Data = df.loc[df['clusters'] == 2];
    cluster4Data = df.loc[df['clusters'] == 3];
    cluster5Data = df.loc[df['clusters'] == 4];
    # print(type(cluster2Data))
    sampleFromCluster1 = cluster1Data.sample(frac=0.50)
    sampleFromCluster2 = cluster2Data.sample(frac=0.50)
    sampleFromCluster3 = cluster3Data.sample(frac=0.50)
    sampleFromCluster4 = cluster4Data.sample(frac=0.50)
    sampleFromCluster5 = cluster5Data.sample(frac=0.50)
    finalSample = sampleFromCluster1.append(sampleFromCluster2).append(sampleFromCluster3).append(
        sampleFromCluster4).append(sampleFromCluster5);
    finalSample = finalSample.drop(columns="clusters")

    # PCA - Task 2
    pca = PCA(n_components=20)
    components = pca.fit_transform(finalSample)
    print(len(pca.components_[0]))
    variance = []
    sum = 0
    for index, val in enumerate(pca.explained_variance_):
        variance.append((index + 1, val))
        # sum += pca.explained_variance_ratio_[index]
    # print(sum)
    '''pdf = pd.DataFrame(data=(pca.components_),\
                       columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10',\
                                'PCA11', 'PCA12', 'PCA13', 'PCA14' \
                           , 'PCA15', 'PCA16', 'PCA17', 'PCA18', 'PCA19', 'PCA20'])'''
    #print(pdf)
    data = pd.DataFrame(variance, columns=list("ki"))
    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata}
    return render_template("pca1.html", data=data)

@app.route("/mds", methods=['POST', 'GET'])
def mds():

    global df
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df))
    kmeans = KMeans(n_clusters=5, random_state=1)
    kmeans.fit(normalized_df)
    # print(kmeans.labels_)
    normalized_df['clusters'] = kmeans.labels_
    cluster1Data = normalized_df.loc[normalized_df['clusters'] == 0];
    cluster2Data = normalized_df.loc[normalized_df['clusters'] == 1];
    cluster3Data = normalized_df.loc[normalized_df['clusters'] == 2];
    cluster4Data = normalized_df.loc[normalized_df['clusters'] == 3];
    cluster5Data = normalized_df.loc[normalized_df['clusters'] == 4];
    # print(type(cluster2Data))
    sampleFromCluster1 = cluster1Data.sample(frac=0.50)
    sampleFromCluster2 = cluster2Data.sample(frac=0.50)
    sampleFromCluster3 = cluster3Data.sample(frac=0.50)
    sampleFromCluster4 = cluster4Data.sample(frac=0.50)
    sampleFromCluster5 = cluster5Data.sample(frac=0.50)
    finalSample = sampleFromCluster1.append(sampleFromCluster2).append(sampleFromCluster3).append(
        sampleFromCluster4).append(sampleFromCluster5);

    if request.method == 'POST':
        if request.form['data'] == 'before':
            euc_dist = pairwise_distances(normalized_df, metric='euclidean')
            km = []
            # Training K
            for k in range(2,4):
                md = MDS(n_components=k, dissimilarity='euclidean')
                components = md.fit(euc_dist)
                km.append((k, md.stress_ / 100000))

            data = pd.DataFrame(data=km, columns=list("ki"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data) # Should be a json string
        elif request.form['data'] == 'after':
            # After training
            euc_dist = pairwise_distances(finalSample, metric='euclidean')
            km = []
            # Training K
            for k in range(2, 4):
                md = MDS(n_components=k, dissimilarity='euclidean')
                components = md.fit(euc_dist)
                km.append((k, md.stress_ / 100000))

            data = pd.DataFrame(data=km, columns=list("ki"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data)


    euc_dist = pairwise_distances(finalSample,metric='euclidean')
    km = []
    # Training K
    for k in range(2, 4):
        md = MDS(n_components=k,dissimilarity='euclidean')
        components = md.fit(euc_dist)
        km.append((k, md.stress_/100000))

    data = pd.DataFrame(data=km, columns=list("ki"))
    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata}
    return render_template("mds.html", data=data)

if __name__ == "__main__":
    df = pd.read_csv('TEST.csv')
    app.run(debug=True)
