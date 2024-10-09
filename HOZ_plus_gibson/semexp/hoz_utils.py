import h5py
import numpy as np
import semexp.my_cluster as mc
import networkx as nx
import scipy.io as scio
from semexp.km_match import KMMatcher
import cv2
from PIL import Image
import torch

block_size = 50
step_size = 2

# color_pal = [
#     255,192,203,
#     255,105,180,
#     221,160,221,
#     100,149,237,
#     95,158,160,
#     0,250,154,
#     255,250,205,
#     255,160,122,
# ]


def vis_map(vis_map, save_dir):
    img = Image.new("P", (vis_map.shape[1], vis_map.shape[0]))
    img.putpalette(color_pal)
    img.putdata((vis_map.flatten()).astype(np.uint8))
    img = img.convert("RGB")
    img = np.array(img)
    cv2.imwrite(save_dir, img)
    print('debug visual semmap')

def map_conversion(sem_map):
    output_map = np.zeros((17, sem_map.shape[0], sem_map.shape[1]))
    for i in range(1,18): 
        loc = np.where(sem_map==i)
        if len(loc[0]) == 0:
            continue
        if i == 1:
            output_map[0,loc[0],loc[1]] = 1
        elif i == 2:
            output_map[1,loc[0],loc[1]] = 1
        else:
            output_map[i-1,loc[0],loc[1]] = 1
    return output_map

def get_location(index, location_list):
    return_list = []
    for i in index:
        return_list.append(location_list[i])

    return return_list

def get_distance(location_A, location_B):
    count = 0
    for A_point in location_A:
        for B_point in location_B:
            d = abs(float(A_point.split('|')[0])-float(B_point.split('|')[0])) \
                + abs(float(A_point.split('|')[1])-float(B_point.split('|')[1]))
            if d <= 0.5:
                count += 1
    if len(location_A)*len(location_B)==0:
        return 0
    return float(count/(len(location_A)*len(location_B)))

def add_node(G, center_feature, location_list, index):
    num = center_feature.shape[0]
    for i in range(num):
        coordinate_list = get_location(index[i], location_list)
        G.add_node(i, feature=center_feature[i], coordinate=coordinate_list)
    return G

def add_edge(G, center_feature):
    num = center_feature.shape[0]
    for i in range(num):
        for j in range(i+1, num):
            distance = get_distance(G._node[i]['coordinate'], G._node[j]['coordinate'])
            G.add_edge(i, j, weight=distance)

    return G

def cluster_feature(feature_list, zone_number=16):

    record,centers=mc.k_means(feature_list,zone_number,300)

    return record, centers

def get_det(id):
    my_list = []
    location_list = []
    # f = h5py.File("/home/sxz/yxy/HOZ_Graph/data/FloorPlan"+str(id)+"/det_feature_22_cates.hdf5", "r")
    
    # yxy added gibson
    f = h5py.File("/home/sxz/yxy/PONI/data/semantic_maps/gibson/semantic_maps/{}.h5".format(id), "r")
    feature = map_conversion(f['0/map_semantic'][()]) # 17 * h * w
    feature = feature[2:,:,:]
    c, h, w = feature.shape
    p_h = 0
    p_w = 0
    i,j = 0, 0
    while (p_h<h and p_w<w):
        p_h1 = p_h + block_size
        p_w1 = p_w + block_size
        loc = "{}|{}".format(i,j)
        if (p_h1>=h and p_w1>=w):
            tmp_f = feature[:, h-1-block_size:h-1, w-1-block_size:w-1]
            p_w += step_size
            p_h += step_size
        elif p_h1>=h:
            tmp_f = feature[:, h-1-block_size:h-1, p_w:p_w1]
            p_w += step_size
            j += 1
        elif p_w1>=w:
            tmp_f = feature[:, p_h:p_h1, w-1-block_size:w-1]
            p_h += step_size
            p_w = 0
            i += 1
            j = 0
        else:
            tmp_f = feature[:, p_h:p_h1, p_w:p_w1]
            p_w += step_size
            j += 1
        tmp_f = tmp_f.reshape(c, block_size*block_size) # 17 * _
        tmp_1d = np.mean(tmp_f, axis=1)
        tmp_1d = np.squeeze(tmp_1d)
        tmp_1d[tmp_1d>0] = 1
        my_list.append(np.squeeze(tmp_1d))
        location_list.append(loc)  
    
    f.close()
    return my_list, location_list

def get_room_graph(id, zone_number):
    feature_list, location_list = get_det(id)
    cluster_record, center_feature = cluster_feature(feature_list, zone_number=zone_number)

    # region
    # yxy test cluster
    # n=np.zeros((145,154))
    # for i in range(8):
    #     clu = cluster_record[i]
    #     for j in range(len(clu)):
    #         x=int(location_list[clu[j]].split('|')[0])
    #         y=int(location_list[clu[j]].split('|')[1])
    #         n[int(x),int(y)]=i
    # vis_map(n,'/home/sxz/yxy/HOZ_Graph/cluster.jpg')
    # endregion
    
    g = nx.Graph()
    g = add_node(g, center_feature, location_list, cluster_record)
    g = add_edge(g, center_feature)

    return g

def get_weights(vec1, vec2):
    weights = np.zeros((vec1.shape[0], vec2.shape[0]))
    for i in range(vec1.shape[0]):
        for j in range(vec2.shape[0]):
            d = 1/(np.sum(vec1[i]*vec2[j])+0.1)+np.linalg.norm(vec1[i] - vec2[j])
            weights[i][j] = 1.0/d

    return weights

def get_edge_weight(node_list1, node_list2, data):
    edge_weight = []
    for i in range(len(data)):
        edge_weight.append(data[i]['edges'][node_list1[i]][node_list2[i]])
    new_edge_weight = np.mean(np.array(edge_weight))
    return new_edge_weight

def get_scene_graph(data, link):
    new_node_features = []
    for node_link in link:
        node = []
        for index, id in enumerate(node_link):
            node.append(data[index]['node_features'][id])
        new_node_features.append(np.mean(np.array(node), axis=0))
    new_edges = np.ones(data[0]['edges'].shape)
    for i in range(data[0]['edges'].shape[0]):
        for j in range(data[0]['edges'].shape[1]):
            if i == j:
                continue
            else:
                edge_weight = get_edge_weight(link[i], link[j], data)
                new_edges[i][j] = edge_weight

    save_dict = dict(node_features=np.array(new_node_features),
                     edges=np.array(new_edges))
    return save_dict

def change(G):
    features = []

    for k, v in G._node.items():
        features.append(v['feature'])
    edges = np.ones((len(features), len(features)))
    for k_i, v in G._adj.items():
        for k_j, distance in v.items():
            edges[k_i][k_j] = distance['weight']

    save_dict = dict(node_features=np.array(features),
                     edges=np.array(edges))
    return save_dict

def cal_dist(va, vb):
    return 1/(np.sum(va*vb)+0.1) + np.linalg.norm(va-vb) 

def cal_target_id(idx, G):
    feature = np.zeros(G['node_features'].shape[1])
    feature[idx] = 1
    tmp_dist = 1
    tmp_idx = -1
    for i in range(G['node_features'].shape[0]):
        tmp = cal_dist(feature, G['node_features'][i])
        if tmp < tmp_dist:
            tmp_dist = tmp
            tmp_idx = i
    return tmp_idx

def cal_local_map_hoz(G, local_map, agent_loc, idx, area_size=10):
    location_shift = [
        (0,0),
        (0,480),
        (480,0),
        (480,480),
    ]
    
    target_node_id = cal_target_id(idx, G)
    
    areas_dist2target = np.zeros(4)
    
    for i in range(4):
        x1,x2,y1,y2 = min(location_shift[i][0], agent_loc[0]), max(location_shift[i][0], agent_loc[0]), min(location_shift[i][1], agent_loc[1]), max(location_shift[i][1], agent_loc[1])
        tmp = local_map[:, x1:x2, y1:y2]
        tmp = tmp.reshape(local_map.shape[0], (x2-x1) * (y2-y1)).cpu().numpy()
        tar_feat = np.mean(tmp, axis=1)
        tar_feat = np.squeeze(tar_feat)
        # tar_feat[tar_feat>0] = 1
        tar_feat = np.squeeze(tar_feat)
        
        dist = np.zeros(G['node_features'].shape[0])
        for j in range(G['node_features'].shape[0]):
            node = G['node_features'][j]
            dist[j] = cal_dist(tar_feat, node)
        
        id_node = dist.argmin()
        areas_dist2target[i] = G['edges'][id_node][target_node_id]
    
    id_area = areas_dist2target.argmax()
    
    return id_area