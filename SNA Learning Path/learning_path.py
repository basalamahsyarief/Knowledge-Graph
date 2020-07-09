import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import dgl
from dgl.contrib.data import load_data
import pandas as pd
import numpy as np
import os
from model_arch import LinkPredict
from utils import *
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')


class Learning_Path:
    def __init__(self):
        self.model = None
        data = pd.read_csv(APP_STATIC+'/sample_dataset_learnavi.csv',
                           index_col='Unnamed: 0')
        source = data.source_name
        target = data.target_name
        self.num_nodes = len(source.append(target).unique())
        self.num_rels = len(data.edge.unique())
        self.train_data = data[['source', 'edge', 'target']].values
        self.load_model(APP_STATIC+'/model_state.pth')
        self.embed = self.build_graph()

    def load_model(self, state_path):
        state = torch.load(state_path)
        self.model = LinkPredict(self.num_nodes,
                                 140,
                                 self.num_rels,
                                 num_bases=140,
                                 num_hidden_layers=2,
                                 dropout=0.2,
                                 use_cuda=-1,
                                 reg_param=0.01)
        self.model.load_state_dict(state['state_dict'])

    def build_graph(self):
        test_graph, test_rel, test_norm = build_test_graph(
            self.num_nodes, self.num_rels, self.train_data)
        test_deg = test_graph.in_degrees(
            range(test_graph.number_of_nodes())).float().view(-1, 1)
        test_node_id = torch.arange(0, self.num_nodes,
                                    dtype=torch.long).view(-1, 1)
        test_rel = torch.from_numpy(test_rel)
        test_norm = to_edge_norm(test_graph,
                                 torch.from_numpy(test_norm).view(-1, 1))
        embed = self.model(test_graph, test_node_id, test_rel, test_norm)
        return embed

    def predict(self, input):
        input = torch.LongTensor(input).view(-1, 3)
        result = calc_raw_mrr(self.embed, self.model.w_relation,
                              input, [1, 3, 10], 100)
        return result
