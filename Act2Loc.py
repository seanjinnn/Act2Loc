from WUREPR import *
import skmob
import os
import pickle
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from skmob.preprocessing import filtering, compression, detection, clustering
import multiprocessing
import numpy as np
from math import sqrt, sin, cos, pi, asin, pow, ceil
import datetime

def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...

def UO_od_matrix(spatial_tessellation, alpha=0, beta=1, M=0):
    print('Computing origin-destination matrix via radiation model\n')

    ## 参数使用 beta 构建OD-matrx
    n = len(spatial_tessellation)
    od_matrix = np.zeros( (n, n) )

    for id_i in tqdm(spatial_tessellation):  # original
        lat_i, lng_i, m_i = spatial_tessellation[id_i]['lat'], spatial_tessellation[id_i]['lng'], \
                            spatial_tessellation[id_i]['relevance']

        edges = []
        probs = []

        # compute the normalization factor
        normalization_factor = 1.0 / (1.0 - m_i / M)
        #         normalization_factor = 1.0

        destinations_and_distances = []
        for id_j in spatial_tessellation:  # destination
            if id_j != id_i:
                lat_j, lng_j, d_j = spatial_tessellation[id_j]['lat'], spatial_tessellation[id_j]['lng'], \
                                    spatial_tessellation[id_j]['relevance']
                destinations_and_distances += \
                    [(id_j, earth_distance((lat_i, lng_i), (lat_j, lng_j)))]

        # sort the destinations by distance (from the closest to the farthest)
        destinations_and_distances.sort(key=operator.itemgetter(1))

        sij = 0.0
        for id_j, _ in destinations_and_distances:  # T_{ij} = O_i \\frac{1}{1 - \\frac{m_i}{M}}\\frac{m_i m_j}{(m_i + s_{ij})(m_i + m_j + s_{ij})}.
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

        # normalization by row
        sum_odm = sum(od_matrix[id_i])     # free constrained
        if sum_odm > 0.0:
            od_matrix[id_i] /= sum_odm  # balanced factor

    return od_matrix

# 人口分布数据选择：
def load_spatial_tessellation(tessellation, model='S-EPR'):
    # relevance: population
    M = 0
    spatial_tessellation = {}
    f = np.array(tessellation)

    for line in f:
        i = int(line[0])
        if model == 'S-EPR':
            relevance = 1
        else:
            relevance = int(line[3])

        spatial_tessellation[i] = {'lat': float(line[1]),
                                    'lng': float(line[2]),
                                    'relevance': round(relevance)}

        M += relevance

    return spatial_tessellation, M


def generating_mobility(trajectory, start_date, uid, spatial_tessellation):
    current_date = start_date
    V = trajectory

    uid_list, date_list, tile_list, lat_list, lng_list = [], [], [], [], []
    for v in V:
        uid_list.append(uid)
        date_list.append(current_date)
        tile_list.append(v)
        lat_list.append(spatial_tessellation[v]['lat'])
        lng_list.append(spatial_tessellation[v]['lng'])
        # D.append([uid, current_date, v, spatial_tessellation[v]['lat'], spatial_tessellation[v]['lon']])
        current_date += datetime.timedelta(hours=1)   # datetime compute
    return uid_list, date_list, tile_list, lat_list, lng_list

def generating_traj(trajectory, spatial_tessellation, start_date):
    user = len(trajectory)
    current_date = start_date

    uid_list, date_list, tile_list, lat_list, lng_list = [], [], [], [], []
    for uid in tqdm(np.arange(user)):
        # generate mobility diary
        uid_, date_, tile_, lat_, lng_ = generating_mobility(trajectory[uid], current_date, uid, spatial_tessellation)

        uid_list.extend(uid_)
        date_list.extend(date_)
        tile_list.extend(tile_)
        lat_list.extend(lat_)
        lng_list.extend(lng_)

    traj = pd.DataFrame()
    traj["uid"] = uid_list
    traj["datetime"] = date_list
    traj["cluster"] = tile_list
    traj["lat"] = lat_list
    traj["lng"] = lng_list

    return traj

def run():

    createVar = locals()

    locations = ["shenzhen", "wuhan", "shanghai"] # "NYC", "TKY", "london",
    models = ['WRU-EPR']

    for location in locations:
        tessellation = gpd.read_file(
            "C:/Users/86152/Desktop/tessellation/" + location + "/" + location + "_1km/" + location + ".shp")

        if location == 'shenzhen':
            start_date = pd.to_datetime('2021-11-01 00:00:00')
            x, y = 0.01, 0.16
            beta, tau = 1.00, 0.0128
            days = 30
            nums = 50000
            type = "Mobile"
        elif location == 'wuhan':
            start_date = pd.to_datetime('2019-10-01 00:00:00')
            x, y = 0.00, 0.08
            beta, tau = 1.000, 0.0131
            days = 31
            nums = 50000
            type = "Mobile"
        elif location == 'shanghai':
            start_date = pd.to_datetime('2023-11-01 00:00:00')
            x, y = 0.00, 0.08
            beta, tau = 1.000, 0.0174
            days = 30
            nums = 50000
            type = "Mobile"
        elif location == 'NYC':
            start_date = pd.to_datetime('2012-04-01 00:00:00')
            x, y = 0.00, 0.12
            beta, tau = 1.000, 0.0037
            days = 30+31+30+31+31+30+31+30+31
            nums = 1083
            type = 'Check_in'
        elif location == 'TKY':
            start_date = pd.to_datetime('2012-04-01 00:00:00')
            x, y = 0.00, 0.12
            beta, tau = 1.000, 0.0037
            days = 30+31+30+31+31+30+31+30+31
            nums = 2293
            type = 'Check_in'
        elif location == 'london':
            start_date = pd.to_datetime('2012-08-01 00:00:00')
            x, y = 0.00, 0.12
            beta, tau = 1.000, 0.0015
            days = 31+30+31+30+31
            nums = 11435
            type = 'Check_in'

        for model in models:
            file_path = location+"_od/"+location+"_" + str(round(x, 2)) + "_" + str(round(y, 2)) + ".pkl"
            spatial_tessellation, M = load_spatial_tessellation(tessellation, model=location)

            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    od_matrix = pickle.load(f)
            else:
                od_matrix = UO_od_matrix(spatial_tessellation, alpha=x, beta=y, M=M)
                with open(file_path, 'wb') as f:
                    pickle.dump(od_matrix, f)

            trajectory_generator = WUR_EPR(beta, tau, type)
            trajectory = []
            for i in tqdm(range(nums)):
                synthetic_trajectory = trajectory_generator.move(spatial_tessellation=spatial_tessellation,
                                                                 od_matrix=od_matrix, days=days)
                trajectory.append(synthetic_trajectory)

            tdf = generating_traj(trajectory, spatial_tessellation, start_date)
            tdf = skmob.TrajDataFrame(tdf)
            tdf = compression.compress(tdf)

        with open(location + "_result/" + model + ".pkl", 'wb') as f:
            pickle.dump(tdf, f)
        f.close()

if __name__ == '__main__':

    run()

