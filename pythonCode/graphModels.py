#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

#Parts of code in this file have been taken (copied) from DeepChem (https://github.com/deepchem/)
#Copyright 2017 PandeLab

import collections
import numpy as np
import six
import tensorflow as tf

import deepchem.feat
import deepchem.feat.mol_graphs
import deepchem.metrics
import deepchem.models
import deepchem.models.tensorgraph
import deepchem.models.tensorgraph.layers

from deepchem.data import NumpyDataset
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract, MessagePassing, SetGather
from deepchem.models.tensorgraph.graph_layers import WeaveLayerFactory
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, GraphConv, BatchNorm, GraphPool, GraphGather, WeightedError, Dropout, BatchNormalization, Stack, Flatten, GraphCNN, GraphCNNPool
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms
from multiprocessing.pool import ThreadPool



convFeat=deepchem.feat.ConvMolFeaturizer()
weaveFeat=deepchem.feat.WeaveFeaturizer()

def convFunc(x):
  return convFeat([x])[0]

def weaveFunc(x):
  return weaveFeat([x])[0]

def convInput(model, X_b):
  d = {}
  multiConvMol = deepchem.feat.mol_graphs.ConvMol.agglomerate_mols(X_b)
  d[model.atom_features] = multiConvMol.get_atom_features()
  d[model.degree_slice] = multiConvMol.deg_slice
  d[model.membership] = multiConvMol.membership
  for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
    d[model.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
  return d

def weaveInput(model, X_b):
  feed_dict = dict()
  atom_feat = []
  pair_feat = []
  atom_split = []
  atom_to_pair = []
  pair_split = []
  start = 0
  for im, mol in enumerate(X_b):
    n_atoms = mol.get_num_atoms()
    atom_split.extend([im] * n_atoms)
    C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
    atom_to_pair.append(np.transpose(np.array([C1.flatten() + start, C0.flatten() + start])))
    pair_split.extend(C1.flatten() + start)
    start = start + n_atoms
    atom_feat.append(mol.get_atom_features())
    pair_feat.append(np.reshape(mol.get_pair_features(), (n_atoms * n_atoms, model.n_pair_feat)))

  feed_dict[model.atom_features] = np.concatenate(atom_feat, axis=0)
  feed_dict[model.pair_features] = np.concatenate(pair_feat, axis=0)
  feed_dict[model.pair_split] = np.array(pair_split)
  feed_dict[model.atom_split] = np.array(atom_split)
  feed_dict[model.atom_to_pair] = np.concatenate(atom_to_pair, axis=0)
  return feed_dict



class MySigmoid(deepchem.models.tensorgraph.layers.Layer):
  def __init__(self, in_layers=None, **kwargs):
    super(MySigmoid, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent = inputs[0]
    out_tensor = tf.nn.sigmoid(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class MySigmoidCrossEntropy(deepchem.models.tensorgraph.layers.Layer):
  def __init__(self, in_layers=None, **kwargs):
    super(MySigmoidCrossEntropy, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[1].shape[:-1]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    labels, logits = inputs[0], inputs[1]
    out_tensor =  tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class MyBatchNorm(deepchem.models.tensorgraph.layers.Layer):
  def __init__(self, in_layers=None, **kwargs):
    super(MyBatchNorm, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    #istraining = inputs[1]
    out_tensor = tf.layers.batch_normalization(parent_tensor, training=kwargs['training']>0.5)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor



class MyWeaveTensorGraph(TensorGraph):

  def __init__(self, n_tasks, n_atom_feat=75, n_pair_feat=14, n_hidden=[50], n_graph_feat=[128], dropout=0.0, mode="classification", **kwargs):
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    self.dropout = dropout
    self.mode = mode
    super(MyWeaveTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.pair_features = Feature(shape=(None, self.n_pair_feat))
    self.pair_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
    weave_layer1A, weave_layer1P = WeaveLayerFactory(n_atom_input_feat=self.n_atom_feat, n_pair_input_feat=self.n_pair_feat, n_atom_output_feat=self.n_hidden[0], n_pair_output_feat=self.n_hidden[0], in_layers=[self.atom_features, self.pair_features, self.pair_split, self.atom_to_pair])
    for myind in range(1, len(self.n_hidden)-1):
      weave_layer1A, weave_layer1P = WeaveLayerFactory(n_atom_input_feat=self.n_hidden[myind-1], n_pair_input_feat=self.n_hidden[myind-1], n_atom_output_feat=self.n_hidden[myind], n_pair_output_feat=self.n_hidden[myind], update_pair=True, in_layers=[weave_layer1A, weave_layer1P, self.pair_split, self.atom_to_pair])
    if len(self.n_hidden)>1.5:
      myind=len(self.n_hidden)-1
      weave_layer1A, weave_layer1P = WeaveLayerFactory(n_atom_input_feat=self.n_hidden[myind-1], n_pair_input_feat=self.n_hidden[myind-1], n_atom_output_feat=self.n_hidden[myind], n_pair_output_feat=self.n_hidden[myind], update_pair=False, in_layers=[weave_layer1A, weave_layer1P, self.pair_split, self.atom_to_pair])
    dense1 = Dense(out_channels=self.n_graph_feat[0], activation_fn=tf.nn.tanh, in_layers=weave_layer1A)
    #batch_norm1 = BatchNormalization(epsilon=1e-5, mode=1, in_layers=[dense1])
    batch_norm1 = MyBatchNorm(in_layers=[dense1])
    weave_gather = WeaveGather(self.batch_size, n_input=self.n_graph_feat[0], gaussian_expand=False, in_layers=[batch_norm1, self.atom_split])
    
    weave_gatherBatchNorm2 = MyBatchNorm(in_layers=[weave_gather])
    curLayer=weave_gatherBatchNorm2
    for myind in range(1, len(self.n_graph_feat)-1):
      curLayer=Dense(out_channels=self.n_graph_feat[myind], activation_fn=tf.nn.relu, in_layers=[curLayer])
      curLayer=Dropout(self.dropout, in_layers=[curLayer])
    
    classification = Dense(out_channels=self.n_tasks, activation_fn=None, in_layers=[curLayer])
    sigmoid = MySigmoid(in_layers=[classification])
    self.add_output(sigmoid)
    
    self.label = Label(shape=(None, self.n_tasks))
    all_cost = MySigmoidCrossEntropy(in_layers=[self.label, classification])
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)
    

    self.mydense1=dense1
    self.mybatch_norm1=batch_norm1
    self.myweave_gather=weave_gather
    self.myclassification=classification
    self.mysigmoid=sigmoid
    self.myall_cost=all_cost
    self.myloss=loss



class MyGraphConvTensorGraph(TensorGraph):

  def __init__(self, n_tasks, graph_conv_layers=[64, 64], dense_layer_size=[128], dropout=0.0, mode="classification", **kwargs):
    self.n_tasks = n_tasks
    self.mode = mode
    self.error_bars = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
    self.dense_layer_size = dense_layer_size
    self.dropout = dropout
    self.graph_conv_layers = graph_conv_layers
    kwargs['use_queue'] = False
    super(MyGraphConvTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.atom_features = Feature(shape=(None, 75))
    self.degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    self.membership = Feature(shape=(None,), dtype=tf.int32)

    self.deg_adjs = []
    for i in range(0, 10 + 1):
      deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
      self.deg_adjs.append(deg_adj)
    in_layer = self.atom_features
    for layer_size in self.graph_conv_layers:
      gc1_in = [in_layer, self.degree_slice, self.membership] + self.deg_adjs
      gc1 = GraphConv(layer_size, activation_fn=tf.nn.relu, in_layers=gc1_in)
      batch_norm1 = MyBatchNorm(in_layers=[gc1])
      gp_in = [batch_norm1, self.degree_slice, self.membership] + self.deg_adjs
      in_layer = GraphPool(in_layers=gp_in)
    dense = Dense(out_channels=self.dense_layer_size[0], activation_fn=tf.nn.relu, in_layers=[in_layer])
    batch_norm3 = MyBatchNorm(in_layers=[dense])
    batch_norm3 = Dropout(self.dropout, in_layers=[batch_norm3])
    readout = GraphGather(batch_size=self.batch_size, activation_fn=tf.nn.tanh, in_layers=[batch_norm3, self.degree_slice, self.membership] + self.deg_adjs)

    curLayer=readout
    for myind in range(1, len(self.dense_layer_size)-1):
      curLayer=Dense(out_channels=self.dense_layer_size[myind], activation_fn=tf.nn.relu, in_layers=[curLayer])
      curLayer=Dropout(self.dropout, in_layers=[curLayer])

    classification = Dense(out_channels=self.n_tasks, activation_fn=None, in_layers=[curLayer])
    sigmoid = MySigmoid(in_layers=[classification])
    self.add_output(sigmoid)
    
    self.label = Label(shape=(None, self.n_tasks))
    all_cost = MySigmoidCrossEntropy(in_layers=[self.label, classification])
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)
    
    self.mydense=dense
    self.myreadout=readout
    self.myclassification=classification
    self.mysigmoid=sigmoid
    self.myall_cost=all_cost
    self.myloss=loss
