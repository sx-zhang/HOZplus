import bz2
import _pickle as cPickle
import torch
import numpy as np
import cv2
import math
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.nn.functional as F
import h5py

object_type = ["chair", "couch", "potted plant", "bed", "toilet", "tv", "dining-table", "oven", "sink", "refrigerator", "book", "clock", "vase", "cup", "bottle"]

scene_pred = [[0.25, 0.25, 0.25, 0.25],[1, 0, 0, 0],[0.25, 0.25, 0.25, 0.25],[0, 0, 0, 1],[0, 0, 1, 0], 
        [0.5, 0, 0, 0.5], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0.5, 0.5, 0], [0, 1, 0, 0],[0.25, 0.25, 0, 0.5], 
        [0.5, 0, 0, 0.5], [0.75, 0, 0, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]

scene_pred = np.array(scene_pred)
scene_pred = torch.from_numpy(scene_pred)
scene_dict = {0:"living room", 1:"kitchen", 2:"bathroom", 3:"bedroom"}

with open("../human_results.txt", "r") as fp:
    tmp = fp.readlines()
    
room_type_search = {}
for line in tmp:
    object_current, room_type = line.strip().split(": ")
    room_type_search[object_current] = room_type

def get_nodes(map, patch_size=80):
    zone_size = 15
    theta = 20
    sem_map = torch.from_numpy(map)
    room_size = 0.05*0.05*sem_map.shape[1]*sem_map.shape[2]
    num_zone = max(int(room_size/zone_size+0.5), 1)
    patch_size = int((zone_size/theta)**0.5 / 0.05)
    if patch_size%2==1:patch_size+=1

    pool = nn.MaxPool2d(kernel_size=(patch_size,patch_size), stride=patch_size//2, padding=0)
    patch_map = pool(sem_map)
    r_map = patch_map.reshape(map.shape[0],-1).transpose(1,0) # N*22
    
    kmeans = KMeans(n_clusters=num_zone, random_state=0, max_iter=300).fit(r_map)
    centers = torch.from_numpy(kmeans.cluster_centers_)
    label_map = kmeans.labels_
    label_map = label_map.reshape(patch_map.shape[1],patch_map.shape[2])
    
    for i in range(label_map.shape[0]):
        for j in range(label_map.shape[1]):
            if patch_map[:,i,j].max()==0.0:
                label_map[i][j]=-1
    
    center_ordinates = []
    mean_centers = []
    num_zone = np.max(label_map)+1
    for i in range(num_zone):
        _, _, stats_t, centroids_t = cv2.connectedComponentsWithStats((label_map==i).astype(np.uint8), connectivity=4)
        center_ordinates.append(centroids_t)
        max_id, max_s = -1, 0
        for j in range(stats_t.shape[0]):
            if stats_t[j][4]>max_s:
                max_s = stats_t[j][4]
                max_id = j
        mean_centers.append(centroids_t[max_id])

    graph = []
    scene_label = torch.argmax(torch.matmul(centers, scene_pred), 1)
    for i in range(num_zone):
        room_tmp = scene_dict[scene_label[i].item()]
        graph.append({'label':room_tmp, 'value':centers[i], 'id':i})

    graph = sorted(graph, key=lambda x:x['id'])
        
    return graph, center_ordinates


def get_edges(graph, cordinates):
    n = len(graph)
    mat = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            a, b = graph[i]['id'], graph[j]['id']
            if a==b:mat[a, b]=0
            else:
                d_ = cal_dist(cordinates[a], cordinates[b])
                mat[a, b] = d_
    mat = F.softmax(mat)
    return mat  


def cal_dist(xy_0, xy_1):
    len_0, len_1 = xy_0.shape[0], xy_1.shape[0]
    sum_dist = 0
    for i in range(len_0):
        for j in range(len_1):
            sum_dist += math.sqrt(math.pow(xy_0[i][0]-xy_1[j][0],2)+math.pow(xy_0[i][1]-xy_1[j][1],2))
    sum_dist = sum_dist / float(len_0*len_1)
    return sum_dist
        
        
def get_zone_graph(map):
    graph, center_ordinates = get_nodes(map)
    edge = get_edges(graph, center_ordinates)
    return graph, edge

                    
all_scenes = [
    "Mifflinburg", "Darden", "Collierville", "Hiteman", "Shelbyville", "Tolstoy", "Wainscott", 
    "Leonardo", "Wiconisco", "Benevolence", "Pomaria", "Coffeen", "Ranchester", "Hanson", "Cosmos", "Marstons", 
    "Corozal", "Lakeville", "Merom", "Klickitat", "Pinesdale", "Woodbine", "Allensville", "Lindenwood", "Onaga", 
    "Beechwood", "Markleeville", "Newfields", "Forkland",
]
scenes_use_floor_0 = ['Corozal', 'Ranchester', 'Onaga', 'Lindenwood', 'Allensville', 'Pinesdale', 'Woodbine', 'Lakeville', 'Beechwood', 'Tolstoy']
scenes_use_floor_1 = ['Collierville', 'Darden', 'Markleeville', 'Wiconisco', 'Pomaria', 'Newfields', 'Mifflinburg', 'Merom', 'Klickitat', 'Hiteman', 'Hanson', 'Forkland', 'Cosmos', 'Benevolence', 'Leonardo', 'Wainscott', 'Marstons', 'Shelbyville']
scenes_use_floor_2 = ['Coffeen']

def map_conversion(sem_map, margin=10):
    output_map = np.zeros((17, sem_map.shape[0], sem_map.shape[1]))
    for i in range(1,18):
        loc = np.where(sem_map==i)
        if len(loc[0])==0:
            continue
        if i==1:
            output_map[1, loc[0], loc[1]] = 1
        elif i==2:
            output_map[0, loc[0], loc[1]] = 1
        else:
            output_map[i-1, loc[0], loc[1]] = 1  
    _, xx, yy = np.where(output_map>0.0)
    min_x, min_y = xx.min()-margin, yy.min()-margin
    max_x, max_y = xx.max()+margin, yy.max()+margin
    new_h = max(100, max_x-min_x)
    new_w = max(100, max_y-min_y)
    return output_map[:,min_x:min_x+new_h+1, min_y:min_y+new_w+1]


for scene_id in all_scenes: 
    print(scene_id)
    f = h5py.File("../semantic_maps/gibson/{}.h5".format(scene_id), "r")
    if scene_id in scenes_use_floor_0:
        sem_map = f['0/map_semantic'][()]
    elif scene_id in scenes_use_floor_1:
        sem_map = f['1/map_semantic'][()]
    elif scene_id in scenes_use_floor_2:
        sem_map = f['2/map_semantic'][()]
    sem_map = map_conversion(sem_map)

    G, E = get_zone_graph(sem_map[2:,:,:])

    with bz2.BZ2File("./saved_graphs/{}_graph.pbz2".format(scene_id), "w") as fp:
        cPickle.dump(
            {
                "nodes": G,
                "edges": E,
            },
            fp
        )