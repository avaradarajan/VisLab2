import json
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
import pandas.plotting as plt
import matplotlib.pyplot as mp
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
    global normalized_df
    global stratifiedSample
    global clusterLabels
    global random_sample
    if request.method == 'POST':
        if request.form['data'] == 'pcascree':

            #Task 2 - Part 3

            pca = PCA(n_components=10)
            components = pca.fit_transform(finalSample)
            pdf = pd.DataFrame(data=(pca.components_), \
                               columns=list(df.columns.values), index=['PCA1', 'PCA2', 'PCA3','PCA4', 'PCA5', 'PCA6','PCA7', 'PCA8', 'PCA9', 'PCA10'])
            varr = []
            for index, val in enumerate(pca.explained_variance_):
                varr.append(val)

            pdf = pdf.T
            pdf['Sum Of Squared Loadings'] = pdf.apply(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, axis=1)
            print()
            print(f'****************************************************************************************')
            print()
            print(pdf)
            print()
            print(f'****************************************************************************************')
            print()
            d = dict()
            for col, val in zip(df.columns.values, pdf['Sum Of Squared Loadings'].tolist()):
                d[col] = val
            print(f'****************** Top 3 Attributes with highest PCA Loadings are***********************')
            print(sorted(d, key=d.get, reverse=True)[:3])
            print(f'****************************************************************************************')

            # PCA - Task 2
            pcaOriginalData = PCA(n_components=10)
            pcaRandomSampleData = PCA(n_components=10)
            pcaStratifiedSampleData = PCA(n_components=10)

            originalDataComponents = pcaOriginalData.fit_transform(normalized_df)
            randomSampleDataComponents = pcaRandomSampleData.fit_transform(random_sample)
            stratifiedSampleDataComponents = pcaStratifiedSampleData.fit_transform(finalSample)

            varianceForOriginalData = []
            varianceForRandomSampleData = []
            varianceForStratifiedSampleData = []

            for index, val in enumerate(pcaOriginalData.explained_variance_):
                varianceForOriginalData.append((index + 1, val))

            for index, val in enumerate(pcaRandomSampleData.explained_variance_):
                varianceForRandomSampleData.append((index + 1, val))

            for index, val in enumerate(pcaStratifiedSampleData.explained_variance_):
                varianceForStratifiedSampleData.append((index + 1, val))

            originalData = pd.DataFrame(varianceForOriginalData, columns=["xval","yval"])
            odata = originalData.to_dict(orient='records')
            odata = json.dumps(odata, indent=2)

            randomData = pd.DataFrame(varianceForRandomSampleData, columns=["xval", "yval"])
            rdata = randomData.to_dict(orient='records')
            rdata = json.dumps(rdata, indent=2)

            stratData = pd.DataFrame(varianceForStratifiedSampleData, columns=["xval", "yval"])
            sdata = stratData.to_dict(orient='records')
            sdata = json.dumps(sdata, indent=2)

            columns = json.dumps({"xc": "PCA Components", "yc": "Eigen Values"})
            numparams = json.dumps({"np": 6})
            data = {'plot_data': odata, 'rdata':rdata, 'sdata':sdata,'columns':columns,'nump':numparams}
            return jsonify(data) # Should be a json string
        elif request.form['data'] == 'pcascatter':
            variance = []
            pca = PCA(n_components=2)
            components = pca.fit_transform(finalSample)
            # print(components)
            pdf = pd.DataFrame(data=(components), \
                               columns=["PCA1", "PCA2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            for val, k in zip(d, clusterLabels):
                variance.append((val[0], val[1], k))
            data = pd.DataFrame(variance, columns=["xval","yval","c"])
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            columns = json.dumps({"xc": "PCA 1", "yc": "PCA 2"})
            numparams = json.dumps({"np": 2})
            data = {'plot_data': cdata, 'columns': columns, 'nump': numparams, 'rdata': [], 'sdata': []}
            return jsonify(data)
        elif request.form['data'] == 'mdsscattereuc':
            lst = []
            # Training K
            md = MDS(n_components=2, dissimilarity='euclidean')
            components = md.fit_transform(finalSample)
            pdf = pd.DataFrame(data=components, columns=["MDS1", "MDS2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            for val, k in zip(d, clusterLabels):
                lst.append((val[0], val[1], k))
            data = pd.DataFrame(data=lst, columns=["xval","yval","c"])
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            columns = json.dumps({"xc": "MDS 1", "yc": "MDS 2"})
            numparams = json.dumps({"np": 2})
            data = {'plot_data': cdata, 'columns': columns, 'nump': numparams, 'rdata': [], 'sdata': []}
            return jsonify(data)
        elif request.form['data'] == 'mdsscattercor':
            lst = []
            euc_dist = pairwise_distances(finalSample, metric='correlation')
            # Training K
            md = MDS(n_components=2)
            components = md.fit_transform(euc_dist)
            pdf = pd.DataFrame(data=components, columns=["MDS1", "MDS2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            for val, k in zip(d, clusterLabels):
                lst.append((val[0], val[1], k))
            data = pd.DataFrame(data=lst, columns=["xval","yval","c"])
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            columns = json.dumps({"xc": "MDS 1", "yc": "MDS 2"})
            numparams = json.dumps({"np": 2})
            data = {'plot_data': cdata, 'columns': columns, 'nump': numparams, 'rdata': [], 'sdata': []}
            return jsonify(data)

    #Task 1 - Part 2 - Stratified Sampling
    #Normalization

    # Training K
    elbowPlotData = []
    for k in range(1,15):
        kmeans = KMeans(n_clusters=k,random_state=1)
        kmeans.fit(normalized_df)
        elbowPlotData.append((k,kmeans.inertia_/1000000))
    data = pd.DataFrame(elbowPlotData, columns=["xval","yval"])
    #Reference to find elbow point - https://stackoverflow.com/questions/51762514/find-the-elbow-point-on-an-optimization-curve-with-python
    kn = KneeLocator(data.xval, data.yval, curve='convex', direction='decreasing')
    print()
    print(f'***********************************  K Value  *****************************************************')
    print()
    print(kn.knee)
    print()
    print(f'***************************************************************************************************')
    print()

    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    columns = json.dumps({"xc":"K Values","yc":"WCSS"})
    numparams = json.dumps({"np": 2})
    data = {'plot_data': cdata,'columns':columns,'nump':numparams, 'rdata':[], 'sdata':[]}
    return render_template("index.html",data=data)


@app.route("/strat", methods=['POST', 'GET'])
def strat():

    global df

    if request.method == 'POST':
        if request.form['data'] == 'before':
            scaler = StandardScaler()
            normalized_df = pd.DataFrame(scaler.fit_transform(df))
            # PCA - Task 2
            pca = PCA(n_components=4)
            components = pca.fit_transform(normalized_df)
            print((pca.explained_variance_ratio_ * 100))
            variance = []
            sum = 0
            for index, val in enumerate(pca.explained_variance_):
                variance.append((index + 1, val))
                # sum += pca.explained_variance_ratio_[index]
            # print(sum)
            '''pdf = pd.DataFrame(data=components,
                               columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10',
                                        'PCA11', 'PCA12', 'PCA13', 'PCA14' \
                                   , 'PCA15', 'PCA16', 'PCA17', 'PCA18', 'PCA19', 'PCA20'])'''
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
            pca = PCA(n_components=4)
            components = pca.fit_transform(finalSample)
            print((pca.explained_variance_ratio_*100))
            variance = []
            sum=0
            for index,val in enumerate(pca.explained_variance_):
                variance.append((index+1,val))
                #sum += pca.explained_variance_ratio_[index]
            #print(sum)
            '''pdf = pd.DataFrame(data=components,columns=['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10','PCA11','PCA12','PCA13','PCA14' \
                ,'PCA15','PCA16','PCA17','PCA18','PCA19','PCA20'])'''
            #print(pdf)
            data = pd.DataFrame(variance, columns=list("ki"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data)
        elif request.form['data'] == 'pca2':
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
            cls = finalSample['clusters']
            finalSample = finalSample.drop(columns="clusters")

            #PCA - Task 2
            variance = []
            pca = PCA(n_components=2)
            components = pca.fit_transform(finalSample)
            #print(components)
            pdf = pd.DataFrame(data=(components), \
                               columns=["PCA1","PCA2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            #print(normalized_df['clusters'])
            for val,k in zip(d,cls):
                variance.append((val[0],val[1],k))
            data = pd.DataFrame(variance, columns=list("kic"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data)

    # After training
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df))
    #normalized_df = pd.DataFrame(preprocessing.scale(df))
    # print(normalized_df)
    kmeans = KMeans(n_clusters=2, random_state=1)
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
    finalSample = finalSample.drop(columns="clusters")

    # PCA - Task 2
    #--------------Try y axis being variance to see diff better
    pca = PCA(n_components=4)
    components = pca.fit_transform(finalSample)
    sum = 0
    for index, val in enumerate(pca.explained_variance_ratio_):
        sum += pca.explained_variance_ratio_[index]*100
    print(sum)
    variance = []
    varr = []
    for index, val in enumerate(pca.explained_variance_):
        variance.append((index + 1, val))
        varr.append(val)
        # sum += pca.explained_variance_ratio_[index]
    # print(sum)
    pca = PCA(n_components=3)
    components = pca.fit_transform(finalSample)
    pdf = pd.DataFrame(data=(pca.components_),\
                       columns=list(df.columns.values),index=['PCA1','PCA2','PCA3'])
    varr = []
    for index, val in enumerate(pca.explained_variance_):
        varr.append(val)
    print(varr)

    pdf = pdf.T
    #pdf['Variance'] = varr
    pdf['Sum Of Squared Loadings'] = pdf.apply(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, axis=1)
    print()
    print(f'****************************************************************************************')
    print()
    print(pdf)
    print()
    print(f'****************************************************************************************')
    print()
    d = dict()
    for col,val in zip(df.columns.values,pdf['Sum Of Squared Loadings'].tolist()):
        d[col] = val
    print(f'****************** Top 3 Attributes with highest PCA Loadings are***********************')
    print(sorted(d,key=d.get, reverse=True)[:3])
    print(f'****************************************************************************************')
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
    cls = finalSample['clusters']
    finalSample = finalSample.drop(columns="clusters")

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


    #euc_dist = pairwise_distances(finalSample,metric='euclidean')
    km = []
    # Training K
    for k in range(2, 10):
        md = MDS(n_components=k,dissimilarity='euclidean')
        components = md.fit(finalSample)
        km.append((k, md.stress_/100000))

    data = pd.DataFrame(data=km, columns=list("ki"))
    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata}
    return render_template("mds.html", data=data)

@app.route("/mdsp", methods=['POST', 'GET'])
def mdsplot():

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
    cls = finalSample['clusters']
    finalSample = finalSample.drop(columns="clusters")

    if request.method == 'POST':
        if request.form['data'] == 'euc':
            euc_dist = pairwise_distances(finalSample, metric='euclidean')
            lst = []
            # Training K
            md = MDS(n_components=2, dissimilarity='euclidean')
            components = md.fit_transform(euc_dist)
            pdf = pd.DataFrame(data=components, columns=["MDS1", "MDS2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            for val, k in zip(d, cls):
                lst.append((val[0], val[1], k))
            data = pd.DataFrame(data=lst, columns=list("kic"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data) # Should be a json string
        elif request.form['data'] == 'cor':
            # After training
            euc_dist = pairwise_distances(finalSample, metric='correlation')
            lst = []
            # Training K
            md = MDS(n_components=2, dissimilarity='euclidean')
            components = md.fit_transform(euc_dist)
            pdf = pd.DataFrame(data=components, columns=["MDS1", "MDS2"])
            d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
            for val, k in zip(d, cls):
                lst.append((val[0], val[1], k))
            data = pd.DataFrame(data=lst, columns=list("kic"))
            cdata = data.to_dict(orient='records')
            cdata = json.dumps(cdata, indent=2)
            data = {'chart_data': cdata}
            return jsonify(data)


    euc_dist = pairwise_distances(finalSample,metric='euclidean')
    lst = []
    # Training K
    md = MDS(n_components=2,dissimilarity='euclidean')
    components = md.fit_transform(euc_dist)
    print(components)
    pdf = pd.DataFrame(data=components,columns=["MDS1", "MDS2"])
    d = pdf.apply(lambda x: (x[0], x[1]), axis=1)
    for val, k in zip(d, cls):
        lst.append((val[0], val[1], k))

    data = pd.DataFrame(data=lst, columns=list("kic"))
    cdata = data.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata}
    return render_template("mdsp.html", data=data)

@app.route("/spm", methods=['POST', 'GET'])
def spm():

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
    cls = finalSample['clusters']
    finalSample = finalSample.drop(columns="clusters")
    finalSample.columns = df.columns.values
    finalDF = finalSample[['batterypower','threeg','fourg']]
    print(finalDF)
    #plt.scatter_matrix(finalDF,alpha=0.3,figsize=(3, 3), diagonal='kde')
    #mp.show()
    #finalDF.columns=["a",""]
    cdata = finalDF.to_dict(orient='records')
    cdata = json.dumps(cdata, indent=2)
    data = {'chart_data': cdata,'d':'hi'}

    return render_template("spmnet.html", data=data)

if __name__ == "__main__":
    df = pd.read_csv('TEST.csv')

    #Standardize data
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df))

    # TASK 1 - Part 1 - Random Sampling
    random_sample = normalized_df.sample(n=500)
    print()
    print(f'***********************************RANDOM SAMPLE*****************************************************')
    print()
    print(random_sample)
    print()
    print(f'*****************************************************************************************************')
    print()
    kmeans = KMeans(n_clusters=5, random_state=31)
    kmeans.fit(normalized_df)
    normalized_df['clusters'] = kmeans.labels_
    cluster1Data = normalized_df.loc[normalized_df['clusters'] == 0];
    cluster2Data = normalized_df.loc[normalized_df['clusters'] == 1];
    cluster3Data = normalized_df.loc[normalized_df['clusters'] == 2];
    cluster4Data = normalized_df.loc[normalized_df['clusters'] == 3];
    cluster5Data = normalized_df.loc[normalized_df['clusters'] == 4];
    sampleFromCluster1 = cluster1Data.sample(frac=0.50)
    sampleFromCluster2 = cluster2Data.sample(frac=0.50)
    sampleFromCluster3 = cluster3Data.sample(frac=0.50)
    sampleFromCluster4 = cluster4Data.sample(frac=0.50)
    sampleFromCluster5 = cluster5Data.sample(frac=0.50)
    stratifiedSample = sampleFromCluster1.append(sampleFromCluster2).append(sampleFromCluster3).append(
        sampleFromCluster4).append(sampleFromCluster5);
    clusterLabels = stratifiedSample['clusters']
    finalSample = stratifiedSample.drop(columns="clusters")

    app.run(debug=True)
