import torch
import numpy as np
import bz2
import _pickle as cPickle
from scipy.optimize import linear_sum_assignment

all_scenes = [
    "Mifflinburg", "Darden", "Collierville", "Hiteman", "Shelbyville", "Tolstoy", "Wainscott", 
    "Leonardo", "Wiconisco", "Benevolence", "Pomaria", "Coffeen", "Ranchester", "Hanson", "Cosmos", "Marstons", 
    "Corozal", "Lakeville", "Merom", "Klickitat", "Pinesdale", "Woodbine", "Allensville", "Lindenwood", "Onaga", 
    "Beechwood", "Markleeville", "Newfields", "Forkland",
]
scenes_use_floor_0 = ['Corozal', 'Ranchester', 'Onaga', 'Lindenwood', 'Allensville', 'Pinesdale', 'Woodbine', 'Lakeville', 'Beechwood', 'Tolstoy']
scenes_use_floor_1 = ['Collierville', 'Darden', 'Markleeville', 'Wiconisco', 'Pomaria', 'Newfields', 'Mifflinburg', 'Merom', 'Klickitat', 'Hiteman', 'Hanson', 'Forkland', 'Cosmos', 'Benevolence', 'Leonardo', 'Wainscott', 'Marstons', 'Shelbyville']
scenes_use_floor_2 = ['Coffeen']
 
label_choices = ['kitchen', 'living room', 'bedroom', 'bathroom'] 
tensor_dim = 15
    
def add_dummy_nodes_and_edges(graph, max_nodes_per_label, tensor_dim):
    label_counts = {label:0 for label in label_choices}
    for node in graph['nodes']:
        label_counts[node['label']]+=1
    for label in label_choices:
        while label_counts[label]<max_nodes_per_label[label]:
            dummy_node = {
                'label':label,
                'value':torch.zeros(tensor_dim, dtype = torch.float64),
                'id':len(graph['nodes'])
            }
            graph['nodes'].append(dummy_node)
            label_counts[label]+=1
    new_K = len(graph['nodes'])
    new_edge = torch.ones(new_K, new_K)
    new_edge[:graph['edges'].shape[0], :graph['edges'].shape[1]] = graph['edges']
    graph['edges'] = new_edge
    return graph

def cosine_similarity_normalized(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    normalized_cosine_similarity = (cosine_similarity + 1) / 2
    return normalized_cosine_similarity

def match_nodes(graph_1, graph_2):
    cost_matrix = []
    for node_1 in graph_1['nodes']:
        row = []
        for node_2 in graph_2['nodes']:
            if node_1['label'] == node_2['label']:
                if node_1['value'].sum()>0.0 and node_2['value'].sum()>0.0:
                    cost = 1-cosine_similarity_normalized(node_1['value'], node_2['value'])
                else:
                    cost = 1
            else:
                cost = float('inf')
            row.append(cost)
        cost_matrix.append(row)
    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def merge_graph(graph_1, graph_2, row_ind, col_ind, count):
    merged_nodes = []
    for i, j in zip(row_ind, col_ind):
        new_node = {
            'label': graph_1['nodes'][i]['label'],
            'value': (graph_1['nodes'][i]['value'] * count + graph_2['nodes'][j]['value']) / (count + 1),
            'id': len(merged_nodes)
        }
        merged_nodes.append(new_node)
    new_K = len(merged_nodes)
    merged_edges = torch.zeros(new_K, new_K)
    for i in range(new_K):
        for j in range(new_K):
            xxxx = (graph_1['edges'][row_ind[i], row_ind[j]] * count + graph_2['edges'][col_ind[i], col_ind[j]]) / (count + 1)
            if not 0<=xxxx<=1:
                a=1
                print(graph_1['edges'][row_ind[i], row_ind[j]], graph_2['edges'][col_ind[i], col_ind[j]])
            merged_edges[i, j] = (graph_1['edges'][row_ind[i], row_ind[j]] * count + graph_2['edges'][col_ind[i], col_ind[j]]) / (count + 1)
    new_graph = {
        'nodes': merged_nodes,
        'edges': merged_edges
    }
    return new_graph

with bz2.BZ2File("./saved_graphs/{}_graph.pbz2".format(all_scenes[0]), "rb") as fp:
    graph_1 = cPickle.load(fp)
    
for i in range(1, len(all_scenes)):
    with bz2.BZ2File("./saved_graphs/{}_graph.pbz2".format(all_scenes[i]), "rb") as fp:
        graph_2 = cPickle.load(fp)

    max_nodes_per_label = {
        label:max(
            len([node for node in graph_1['nodes'] if node['label']==label]),
            len([node for node in graph_2['nodes'] if node['label']==label])
        )
        for label in label_choices
    }

    graph_1 = add_dummy_nodes_and_edges(graph_1, max_nodes_per_label, tensor_dim)
    graph_2 = add_dummy_nodes_and_edges(graph_2, max_nodes_per_label, tensor_dim)

    row_ind, col_ind = match_nodes(graph_1, graph_2)
    
    merged_graph = merge_graph(graph_1, graph_2, row_ind, col_ind, i)
    graph_1 = merged_graph
    print(i, len(all_scenes))

with bz2.BZ2File("./saved_graphs/merged_gibson_hoz_graph.pbz2", "w") as fp:
    cPickle.dump(
        {
            "nodes": graph_1['nodes'],
            "edges": graph_1['edges']
        },
        fp
    )