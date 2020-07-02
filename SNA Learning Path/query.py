import time
import networkx as nx
import community
import io
import codecs
import json
import os
from datetime import datetime
import certifi
import pandas as pd
import flask
from sklearn.utils import shuffle

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')


class QuerySNA:
    def __init__(self):
        self.G = nx.Graph()
        self.gdata = {'nodes': [], 'links': []}
        self.hashtags = []
        self.accounts = []
        self.gte = "now-30d/d"
        self.lte = "now/d"
        self.keyword = "(masyarakat AND berpenghasilan AND rendah) OR (perumahan AND subsidi) OR (perumahan AND rakyat)"
        self.fsize = 20
        self.landlordsize = 100
        self.laymansize = 50
        self.tweet_counts = 0

    def qparam(self, gte, lte, keyword, lands, lays, fsize):
        self.G = nx.Graph()
        # self.gdata = {'nodes': [], 'links': []}
        self.gte = gte
        self.lte = lte
        self.keyword = keyword
        self.landlordsize = lands
        self.laymansize = lays
        self.freesize = fsize

    def qretweet(self):
        df = pd.read_csv('sample_dataset_learnavi.csv', index_col='Unnamed: 0')
        # df = shuffle(df)
        # df = df[:100].reset_index(drop=True)
        df.drop_duplicates(inplace=True)
        tipe_source = []
        tipe_target = []
        for a, b, c, d in zip(df.source.astype(str), df.source_name,
                              df.target.astype(str), df.target_name):
            tipe_source.append(b[:-len(a)-1])
            tipe_target.append(d[:-len(c)-1])
        df['source_type'] = tipe_source
        df['target_type'] = tipe_target
        node = []
        s = df.to_dict('records')
        for i, a in enumerate(s):
            nodex = {'id': a['source_name'], 'type': a['source_type']}
            node.append(nodex)
            nodex = {'id': a['target_name'], 'type': a['target_type']}
            node.append(nodex)
            edge = {'id': i, 'type': a['edge_name'],
                    'source': a['source_name'], 'target': a['target_name']}
            self.gdata['links'].append(edge)
        self.gdata['nodes'] = node
        return self.gdata

    def graphanalytic(self):
        # graph analytic
        start_time = time.time()
        self.G = nx.readwrite.json_graph.node_link_graph(self.gdata)
        self.graphcentrality()  # calculate centrality
        self.graphattributes()  # set graph attributes
        elapse_time = time.time() - start_time
        print('graph analytic time: ', elapse_time)
        return self.graphtojson()

    def graphcentrality(self):
        self.betweenness = nx.betweenness_centrality(self.G)  # most expensive
        self.closeness = nx.closeness_centrality(self.G)  # 2nd most expensive
        self.communities = community.best_partition(self.G)

    # #set graph attributes from centrality calculation
    def graphattributes(self):
        nx.set_node_attributes(self.G, self.betweenness, 'betweenness')
        nx.set_node_attributes(self.G, self.closeness, 'closeness')
        nx.set_node_attributes(self.G, self.communities, 'modularity')
    #
    # # save to json format

    def graphtojson(self):
        # jdata = nx.readwrite.json_graph.node_link_data(self.G, {'link': 'edges', 'source': 'sources', 'target': 'target', 'id': 'id'})
        # jdata = nx.readwrite.json_graph.node_link_data(self.G)
        # jdata['edges'][0]['id']=1
        data = nx.readwrite.json_graph.node_link_data(
            self.G, {'link': 'links', 'source': 'source', 'target': 'target'})
        jdata = flask.json.dumps(data, ensure_ascii=False, indent=4)
        nx.write_gml(self.G, "testg.gml")  # save to file
        with io.open(os.path.join(APP_STATIC, 'all.json'), 'w', encoding='utf-8') as f:
            # f.write(flask.json.dumps(data, ensure_ascii=False))
            f.write(jdata)

        return jdata

    def drawplot(self, index):
        if index == 0:
            nx.draw_networkx(self.G, with_labels=False)
        elif index == 1:
            nx.draw_shell(self.G)
        elif index == 2:
            nx.draw_kamada_kawai(self.G, node_size=5, linewidths=0.5)
        elif index == 3:
            nx.draw_random(self.G)

    def getgraph(self):
        return self.G


qsna = QuerySNA()
# print(qsna.hashtags(qsna.qretweet()))
# # start_time = time.time()
qsna.qretweet()
# qsna.simqretweet()
# # elapse_time = time.time() - start_time
# # print ('total time: ', elapse_time)
# # print(qsna.gdata['nodes'])
gjson = qsna.graphanalytic()
# with open(qsna.keyword+'.json', 'w') as f:
#     json.dump(json.loads(gjson),f)
