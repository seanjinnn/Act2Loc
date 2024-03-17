#!/usr/bin/python
#coding:utf-8

# 导入所需的库
import geopandas as gpd  # 用于地理数据处理
import pickle  # 用于数据序列化
import skmob  # 用于移动对象数据处理
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
from shapely import wkt  # 用于几何对象处理
from tqdm import tqdm  # 用于进度条显示
import matplotlib.pyplot as plt  # 用于绘图
import matplotlib as mpl  # Matplotlib的基础设置
from collections import defaultdict  # 用于构建字典
import operator  # 用于运算符操作
import random  # 用于生成随机数
from random import random, uniform, choice  # 随机数生成
from skmob.utils.plot import plot_gdf  # 用于绘制地理数据
from skmob.measures.evaluation import common_part_of_commuters  # 用于计算移动对象之间的交集
import warnings  # 用于警告控制
import math  # 数学库
import powerlaw  # 用于幂律分布拟合
from math import sqrt, sin, cos, pi, asin, pow, ceil  # 数学运算

# 忽略警告
warnings.filterwarnings('ignore')

# 1. 加载网格数据
def load_spatial_tessellation(tessellation):
    # relevance: population
    M = 0
    spatial_tessellation = {}
    f = np.array(tessellation)

    for line in f:
        i = int(line[0])
        relevance = int(line[3])
        if relevance == 0:
            relevance += 1
        spatial_tessellation[i] = {'lat': float(line[2]),
                                    'lon': float(line[1]),
                                    'relevance': round(relevance)}

        M += relevance

    return spatial_tessellation, M

# 将数据生成为列表
def generating_list(tdf, days=30):
    user_location = list()
    location = list()
    for i in range(len(tdf)):
        if i % (days*24) == 0 and i != 0:
            user_location.append(location)
            location = list()
        location.append(tdf[i])
    user_location.append(location)
    return user_location

# 计算两点之间的地球距离
def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...

# 2. 计算起始点和目的地的概率矩阵
def radiation_od_matrix(spatial_tessellation, M, alpha=0, beta=1):
    print('Computing origin-destination matrix via radiation model\n')

    n = len(spatial_tessellation)
    od_matrix = np.zeros((n, n))

    for id_i in tqdm(spatial_tessellation):  # original
        lat_i, lng_i, m_i = spatial_tessellation[id_i]['lat'], spatial_tessellation[id_i]['lon'], \
                            spatial_tessellation[id_i]['relevance']

        edges = []
        probs = []

        # 计算归一化因子
        normalization_factor = 1.0 / (1.0 - m_i / M)

        destinations_and_distances = []
        for id_j in spatial_tessellation:  # destination
            if id_j != id_i:
                lat_j, lng_j, d_j = spatial_tessellation[id_j]['lat'], spatial_tessellation[id_j]['lon'], \
                                    spatial_tessellation[id_j]['relevance']
                destinations_and_distances += [(id_j, earth_distance((lat_i, lng_i), (lat_j, lng_j)))]

        # 根据距离排序目的地
        destinations_and_distances.sort(key=operator.itemgetter(1))

        sij = 0.0
        for id_j, _ in destinations_and_distances:
            m_j = spatial_tessellation[id_j]['relevance']

            if (m_i + sij) * (m_i + sij + m_j) != 0:
                prob_origin_destination = normalization_factor * \
                                          ((m_i + alpha * sij) * m_j) / \
                                          ((m_i + (alpha + beta) * sij) * (m_i + (alpha + beta) * sij + m_j))
            else:
                prob_origin_destination = 0

            sij += m_j
            edges += [[id_i, id_j]]
            probs.append(prob_origin_destination)

        probs = np.array(probs)

        for i, p_ij in enumerate(probs):
            id_i = edges[i][0]
            id_j = edges[i][1]
            od_matrix[id_i][id_j] = p_ij

        # 行归一化
        sum_odm = np.sum(od_matrix[id_i])  # free constrained
        if sum_odm > 0.0:
            od_matrix[id_i] /= sum_odm  # balanced factor

    return od_matrix

# 3. Act2Loc 模型
def weighted_random_selection(weights):
    return np.searchsorted(np.cumsum(weights)[:-1], random())

class Act2Loc:
    def __init__(self):
        self.rho = 0.6
        self.gamma = 0.21

        self.beta = 0.8
        self.tau = 17

        self.other = defaultdict(int)

    # 返回频率最高的地点
    def __frequency_return(self):
        index = weighted_random_selection(list(self.location2visits.values()))
        next_location = list(self.location2visits.keys())[index]
        return next_location

    # 优先选择探索未访问过的地点
    def __preferential_exploration(self, current_location):
        next_location = weighted_random_selection(self.od_matrix[current_location])
        while next_location in self.other.values():
            next_location = weighted_random_selection(self.od_matrix[current_location])
        return next_location

    # 选择下一个地点
    def choose_location(self):
        S = len(self.location2visits)  # 已访问地点数

        if S == 0:  # 初始位置为家
            return self.__preferential_exploration(self.home)

        current_location = self.trajectory[-1]
        return self.__preferential_exploration(current_location)

    # 生成轨迹
    def move(self, home=None, work=None, diary_mobility=None, spatial_tessellation=None,
             od_matrix=None, location_set=None, walk_nums=0):

        self.od_matrix = od_matrix
        self.trajectory = []
        self.location2visits = defaultdict(int)
        self.other = defaultdict(int)

        self.diary_mobility = np.array(diary_mobility)
        self.home = home
        self.location_set = location_set

        if "W" in set(self.diary_mobility):
            self.work = work

            self.od_matrix[self.home][self.work] = 0  # 家 -> 工作地
            sum_odm = np.sum(od_matrix[self.home])
            if sum_odm > 0.0:
                self.od_matrix[self.home] /= sum_odm

            self.od_matrix[self.work][self.home] = 0  # 工作地 -> 家
            sum_odm = np.sum(od_matrix[self.work])
            if sum_odm > 0.0:
                self.od_matrix[self.work] /= sum_odm

        i = 0
        while i < len(self.diary_mobility):
            if self.diary_mobility[i] == 'H':
                next_location = self.home
            elif self.diary_mobility[i] == 'W':
                next_location = self.work
            else:
                if len(self.other) == 0 or self.diary_mobility[i] not in self.other.keys():
                    next_location = self.choose_location()
                    self.other[self.diary_mobility[i]] = next_location
                else:
                    next_location = self.other[self.diary_mobility[i]]

            self.trajectory.append(next_location)
            i += 1

        # 根据访问频率排序
        self.frequency = sorted(self.location2visits.items(), key=lambda d: d[1], reverse=True)

        cnt = 0
        self.mobility = []
        for i in self.diary_mobility:
            if i == 'H':
                location = self.home
            elif i == 'W':
                location = self.work
            else:
                location = self.other[i]
            self.mobility.append(location)

        return self.mobility

# 处理轨迹生成
def func(diary_mobility):
    other_set = set()
    for row in range(len(diary_mobility)):
        if diary_mobility[row] != 'H' and diary_mobility[row] != 'W':
            other_set.add(diary_mobility[row])
    other_set = sorted(other_set)

    walk_nums = 0
    if (len(other_set) > 0):
        walk_nums = max(other_set)

    work = None
    location_set = None

    od_matrix = other_matrix
    if 'W' in diary_mobility:
        work = weighted_random_selection(work_matrix[home])

    home_list.append(home)
    work_list.append(work)

    trajectory = trajectory_generator.move(home=home, work=work, diary_mobility=diary_mobility, od_matrix=od_matrix,
                                           location_set=location_set, walk_nums=walk_nums)

    return trajectory

if __name__ == '__main__':
    # 1. 加载空间划分数据
    tessellation = gpd.read_file(r'data\1km-grids\sz_1km.shp')
    spatial_tessellation, M = load_spatial_tessellation(tessellation)

    # 2. 计算概率矩阵
    work_matrix = radiation_od_matrix(spatial_tessellation, M, alpha=0.13, beta=0.61)
    other_matrix = radiation_od_matrix(spatial_tessellation, M, alpha=0.01, beta=0.45)

    # 3. 加载活动类型序列（由于隐私问题，真实数据集未公开，替换activity.pkl为生成数据即可）
    activity = []
    with open("activity.pkl", "rb") as f:
        activity_set = pickle.load(f)

    # 4. 选择个体数量和居住地
    individuals = 1000
    tessellation["pop"] = tessellation["pop"] / (tessellation["pop"].sum() / individuals)
    tessellation["pop"] = tessellation["pop"].astype("int")
    home_df = tessellation[["tile_ID", "pop"]]

    # 5. 生成轨迹
    diary_mobilitys = []
    trajectory_generator = Act2Loc()
    synthetic_trajectory = []
    home_list = []
    work_list = []
    for i in tqdm(range(len(home_df))):
        home = home_df.iloc[i]["tile_ID"]
        flow = home_df.iloc[i]["pop"]

        for item in range(flow):
            diary_mobility = choice(activity_set)
            diary_mobilitys.append(diary_mobility)

        synthetic_trajectory.extend(list(map(func, diary_mobilitys)))
