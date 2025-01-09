import bz2
import _pickle as cPickle
import torch
import numpy as np
import cv2
import math
import skimage.morphology
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.nn.functional as F
  
  
object_type = ['AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp','Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot','RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster']

def find_mat(map, margin=10):
    c, h, w = map.shape
    for i in range(c):
        map[i] = skimage.morphology.erosion(map[i], selem=skimage.morphology.disk(1))
        map[i] = skimage.morphology.dilation(map[i], selem=skimage.morphology.disk(1))
    _, xx, yy = np.where(map>0.0)
    min_x, min_y = xx.min()-margin, yy.min()-margin
    max_x, max_y = xx.max()+margin, yy.max()+margin
    new_h = max(100, max_x-min_x)
    new_w = max(100, max_y-min_y)
    return map[:,min_x:min_x+new_h+1, min_y:min_y+new_w+1]

def get_nodes(map, zone_size=7, theta=20, room_type=None):
    sem_map = torch.from_numpy(map)
    room_size = 0.02*0.02*sem_map.shape[1]*sem_map.shape[2]
    num_zone = max(int(room_size/zone_size+0.5), 1)
    patch_size = int((zone_size/theta)**0.5 / 0.02)
    if patch_size%2==1:
        patch_size+=1

    pool = nn.MaxPool2d(kernel_size=(patch_size,patch_size), stride=patch_size//2, padding=0)
    patch_map = pool(sem_map)
    r_map = patch_map.reshape(map.shape[0],-1).transpose(1,0) # N*22
    
    kmeans = KMeans(n_clusters=num_zone, random_state=0, max_iter=300).fit(r_map)
    centers = torch.from_numpy(kmeans.cluster_centers_)
    label_map = kmeans.labels_
    label_map = label_map.reshape(patch_map.shape[1],patch_map.shape[2])
    
    center_ordinates = []
    num_zone = np.max(label_map)+1
    for i in range(num_zone):
        _, _, _, centroids_t = cv2.connectedComponentsWithStats((label_map==i).astype(np.uint8), connectivity=4)
        center_ordinates.append(centroids_t)
    
    graph = []
    for i in range(num_zone):
        graph.append({'label':room_type, 'value':centers[i], 'id':i})
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
        

def get_zone_graph(map, zone_size=7, theta=20, room_type=None):
    graph, center_ordinates = get_nodes(map, zone_size=zone_size, theta=theta, room_type=room_type)
    edge = get_edges(graph, center_ordinates)
    return graph, edge

                
kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

# scenes = kitchens + living_rooms + bedrooms + bathrooms

run_id = 1
if run_id==1:
    scenes = kitchens
    room_type = "kitchen"
    save_type = "kitchens"
elif run_id==2:
    scenes = living_rooms
    room_type = "living room"
    save_type = "living_rooms"
elif run_id==3:
    scenes = bedrooms
    room_type = "bedroom"
    save_type = "bedrooms"
elif run_id==4:
    scenes = bathrooms
    room_type = "bathroom"
    save_type = "bathrooms"
 
for scene_id in scenes: 
    print(scene_id)  
    with bz2.BZ2File("../semantic_maps/ai2thor/{}/{}.pbz2".format(save_type, scene_id), "rb") as fp:
        sem_map = cPickle.load(fp)['sem_map']
        sem_map = find_mat(sem_map[:, 400:1600, 400:1600])
        G, E = get_zone_graph(sem_map, room_type=room_type)
        with bz2.BZ2File("./saved_graphs/{}_graph.pbz2".format(scene_id), "w") as fp:
            cPickle.dump(
                {
                    "nodes": G,
                    "edges": E,
                },
                fp
            )
