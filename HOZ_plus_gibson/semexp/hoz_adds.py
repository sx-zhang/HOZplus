import numpy as np
import quaternion
import _pickle as cPickle
import bz2
import sys
sys.path.append("..")
import h5py
import json
import os
import math
import skimage.morphology
import cv2
from semexp.envs.utils.fmm_planner import FMMPlanner
import numpy.ma as ma
import scipy.signal as signal
import scipy.io as scio
import torch
from semexp.hoz_utils import get_room_graph, cal_local_map_hoz
from semexp.utils.visualize_tools import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# sxz use gt val map
dataset_info_file = '..data/datasets/objectnav/gibson/v1.1/val/val_info.pbz2'
dataset_file_path = '..data/semantic_maps/gibson/semantic_maps/'
with bz2.BZ2File(dataset_info_file, "rb") as f:
    dataset_info_1 = cPickle.load(f)
with open("..data/semantic_maps/gibson/semantic_maps/semmap_GT_info.json",'r') as fp:
    dataset_info = json.load(fp)

LOCAL_MAP_SIZE = 480  # TO DO
OBJECT_BOUNDARY = 1 - 0.5
MAP_RESOLUTION = 5
# location_shift = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),
#                   (0,2),(1,2),(2,2),(2,1),(2,0),(2,-1),(2,-2),(1,-2),(0,-2),
#                   (-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2)]


def convert_3d_to_2d_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def gt_sem_map(current_episodes):
    # current_episodes = envs.get_current_episodes()[e]
    scene_name = current_episodes['scene_id']
    scene_name = os.path.basename(scene_name).split(".")[0]
    scene_data_file_path = dataset_file_path + scene_name + ".h5"
    goal_idx = current_episodes["object_ids"][0]
    floor_idx = 0
    scene_info = dataset_info_1[scene_name]
    shape_of_gt_map = scene_info[floor_idx]["sem_map"].shape
    f = h5py.File(scene_data_file_path, "r")
    if scene_name=="Corozal":
        sem_map=f['0/map_semantic'][()].transpose()
    else:
        sem_map=f['1/map_semantic'][()].transpose()
        
    w1, h1 = int(sem_map.shape[0]/2), int(sem_map.shape[1]/2)
    w2, h2 = int(shape_of_gt_map[1]/2), int(shape_of_gt_map[2]/2)
    sem_map1 = sem_map[w1-w2:w1+w2,h1-h2:h1+h2]
    central_pos = dataset_info[scene_name]["central_pos"]
    map_world_shift = dataset_info[scene_name]["map_world_shift"]
    map_obj_origin = scene_info[floor_idx]["origin"]
    min_x, min_y = map_obj_origin / 100.0
    pos = current_episodes["start_position"]
    rot = quaternion.from_float_array(current_episodes["start_rotation"])
    x, y, o = convert_3d_to_2d_pose(pos, rot)
    start_x, start_y = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
    sem_map2 = map_conversion(sem_map1, start_x, start_y, o)
    goal_loc = (sem_map2 == goal_idx+5.0)
    return goal_loc, sem_map2

def map_conversion(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi*1 - start_o)
    cos = math.cos(np.pi*1 - start_o)
    for i in range(18): 
        loc = np.where(sem_map==i)
        if len(loc[0]) == 0:
            continue
        a = loc[0] - start_x
        b = loc[1] - start_y
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0]) == 0:
            continue
        if i == 0:
            pass
        elif i == 1:
            color_index = 2
            output_map[loc_conversion] = color_index
        elif i == 2:
            color_index = 1
            output_map[loc_conversion] = color_index
        else:
            color_index = i+2
            output_map[loc_conversion] = color_index
    output_map = signal.medfilt(output_map, 3)
    return output_map

def map_conversion_old(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi - start_o)
    cos = math.cos(np.pi - start_o)
    for i in range(sem_map.shape[0]):
        loc = np.nonzero(sem_map[i])
        a = loc[0] - start_x
        b = loc[1] - start_y
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0]) == 0:
            continue
        if i == 0:
            color_index = i+2.0 
        else:
            color_index = i+4.0
        output_map[loc_conversion] = color_index
    output_map = signal.medfilt(output_map, 3)
    return output_map
    
def void_out_of_boundary(locs):
    new_locs = [[],[]]
    for i in range(locs[0].shape[0]):
        if 0<locs[0][i]<LOCAL_MAP_SIZE and 0<locs[1][i]<LOCAL_MAP_SIZE:
            new_locs[0].append(locs[0][i])
            new_locs[1].append(locs[1][i])
        else:
            continue
    return [np.array(new_locs[0]), np.array(new_locs[1])]

def gt_stg(goal_loc, sem_map, current_loc):
    # distance map
    exp_map = np.zeros_like(sem_map)
    exp_map[sem_map==2] = 1
    # selem = skimage.morphology.disk(5)
    # traversible = cv2.dilate(exp_map, selem)
    planner = FMMPlanner(exp_map, step_size=5)
    selem = skimage.morphology.disk(
        int(OBJECT_BOUNDARY * 100.0 / MAP_RESOLUTION)
    )
    goal_map = np.zeros_like(sem_map)
    goal_map[goal_loc] = 1
    goal_map = cv2.dilate(goal_map, selem)
    planner.set_multi_goal(goal_map, validate_goal=True)
    dist_map = planner.fmm_dist
    # Circle
    circle_o = np.zeros_like(sem_map)
    circle_o[current_loc[0],current_loc[1]] = 1
    circle_o = cv2.dilate(circle_o, skimage.morphology.disk(80))
    mx_circle = ma.masked_array(exp_map, 1-circle_o)
    mx_circle = ma.filled(mx_circle, 0)
    ret, labels = cv2.connectedComponents(mx_circle.astype(np.int8))
    label_of_current_loc = labels[current_loc[0], current_loc[1]]
    if not label_of_current_loc:
        for s in location_shift:
            label_of_current_loc = labels[current_loc[0]+s[0], current_loc[1]+s[1]]
            if label_of_current_loc:
                break
    
    selected_index = (labels == label_of_current_loc)
    selected_map = np.zeros_like(sem_map)
    selected_map[selected_index]=1
    if not label_of_current_loc:
        selected_map = mx_circle
    circle_dist_map = ma.masked_array(dist_map, 1-selected_map)
    circle_dist_map = ma.filled(circle_dist_map, np.argmax(circle_dist_map))
    m = np.argmin(circle_dist_map)
    r, c = divmod(m, circle_dist_map.shape[1])
    
    dist_map_vis = cv2.applyColorMap(circle_dist_map.astype(np.uint8), cv2.COLORMAP_JET)
    dist_map_vis[r-3:r+3, c-3:c+3] = 100
    dist_map_vis[circle_dist_map>2.0] = 1
    
    # cv2.imwrite('..semexp/sxz/img/dist_circle_test1.png',np.flipud(dist_map_vis), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return r, c, selected_map

def gt_stg_v2(goal_loc, sem_map, current_loc):
    goal_map = np.zeros_like(sem_map)
    goal_map[goal_loc==True] = 1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_map.astype(np.int8),connectivity=4,ltype=None)
    centroids = np.rint(centroids,).astype(np.int)
    dist = np.sum(np.square(centroids[1:]-np.array(current_loc)),axis=1)
    target_index = dist.argsort()
    target_locs = centroids[target_index+1]
    # target_locs = target_locs[np.lexsort(target_locs.T)]
    # sem_map_2 = np.zeros_like(sem_map)
    # sem_map_2[sem_map==2] = 1
    return target_locs

def cosVector(x,y):
    if(len(x)!=len(y)):
        # print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]*0.0001 #10000.0
        result2+=(x[i]*0.01)**2 
        result3+=(y[i]*0.01)**2 
    return (result1/((result2*result3)**0.5)) 

def set_target_hoz_frontier(local_map, t_area_pfs, hoz_graph, target_id, save_path, area_size=50, margin=25):
    xx, yy = torch.where(local_map[1]>0.0)
    if len(xx)>0:
        min_x, min_y = max(0, xx.min()-margin), max(0, yy.min()-margin)
        max_x, max_y = min(479, xx.max()+margin), min(479, yy.max()+margin)
    else:
        min_x, min_y = 190,190
        max_x, max_y = 290,290
    if max_x-min_x<100:
        min_x, max_x = (min_x+max_x)//2-50, (min_x+max_x)//2+50
    if max_y-min_y<100:
        min_y, max_y = (min_y+max_y)//2-50, (min_y+max_y)//2+50
    
    pool_for_hoz = nn.MaxPool2d(kernel_size=(area_size,area_size), stride=area_size, padding=0)
    used_map = local_map[4:-1, min_x:max_x+1, min_y:max_y+1]
    patch_list = pool_for_hoz(used_map)
    
    _, x_num, y_num = patch_list.shape
    patch_list = patch_list.reshape(15,-1).transpose(1,0)
    
    target_vec = torch.zeros(1,15).to(used_map.device)
    target_vec[0,target_id] = 1
    target_node_id = F.cosine_similarity(target_vec.unsqueeze(1), hoz_graph['nodes_mat'].unsqueeze(0), dim=2)[0].argmax()

    p_hoz = torch.zeros(patch_list.shape[0]).to(patch_list.device)
    hoz_sim_mat = F.cosine_similarity(patch_list.unsqueeze(1), hoz_graph['nodes_mat'].unsqueeze(0), dim=2).argmax(1) # shape patch_list.shape[0]
    for i in range(hoz_sim_mat.shape[0]):
        p_hoz[i] = 1-hoz_graph['edges'][hoz_sim_mat[i], target_node_id]
        
    pool_for_area = nn.MaxPool2d(kernel_size=(area_size,area_size), stride=area_size, padding=0)
    p_area = pool_for_area(t_area_pfs[min_x:max_x+1, min_y:max_y+1].unsqueeze(0)).squeeze().reshape(-1)
    
    pool_for_path = nn.AvgPool2d(kernel_size=(area_size,area_size), stride=area_size, padding=0)
    p_path = 1-pool_for_path(local_map[3:4, min_x:max_x+1, min_y:max_y+1]).squeeze().reshape(-1)

    p_overall = p_hoz+p_area+p_path
    new_mat = []
    if max(p_overall)>0.0:
        # chosen_id = torch.multinomial(p_overall, 1, replacement=False)[0]
        chosen_id = p_overall.argmax()
        chosen_x, chosen_y = chosen_id // y_num, chosen_id % y_num
        tmp_patch = t_area_pfs[min_x:max_x+1, min_y:max_y+1][chosen_x*area_size:(chosen_x+1)*area_size, chosen_y*area_size:(chosen_y+1)*area_size]
        tmp_point = torch.where(tmp_patch==tmp_patch.max())
        select_point = (tmp_point[0][0]+chosen_x*area_size+min_x, tmp_point[1][0]+chosen_y*area_size+min_y)

        new_pred = p_overall.reshape(x_num, y_num)
        softmax = nn.Softmax()
        new_mat.append(p_overall)
        new_mat.append(p_hoz)
        new_mat.append(p_area)
        new_mat.append(p_path)
    else:
        tmp_point = torch.where(t_area_pfs==t_area_pfs.max())
        select_point = (tmp_point[0][0], tmp_point[1][0])
    return select_point, new_mat
    
    