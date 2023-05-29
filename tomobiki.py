# coding:utf-8

"""
main module of tomobiki
"""

# @InProceedings{
#   新井2014,
#   Title = {クラスタリングと空間分割の併用による効率的なk-匿名化},
#   Author = {新井淳也, 鬼塚真, 塩川浩昭},
#   Booktitle = {日本データベース学会和文論文誌, Vol.13-J, No.1},
#   Year = {2014},
#   Address = {Tokyo, JP},
#   Pages = {72--77},
#   Publisher = {日本データベース学会},
#   ISSN = {2189-0374},
# }


import pandas as pd
import numpy as np
import networkx as nx
import csv
import sys
import random
import itertools
import time
from scipy.spatial import distance
from mondrian import mondrian


# 引数pointsの頂点の集まりの中で頂点target_pointから近いm個の点を取得する関数
def find_m_nearest_points(points, target_point, m):
    distances = distance.cdist([np.array(list(target_point))], np.array(list(points))).flatten()  # ユークリッド距離を計算
    sorted_indices = np.argsort(distances)  # 距離の昇順にソートされたインデックスを取得
    m_nearest_indices = sorted_indices[:m]  # 距離が近いk個のインデックスを取得
    m_nearest_points = [list(points)[i] for i in m_nearest_indices]  # 距離が近いk個の点を取得
    return m_nearest_points

# 友引法における(k, m)-近傍グラフを構築する関数
def make_graph(data, k, m):
    # nxのグラフオブジェクトに頂点を追加するにはイミュータブルなオブジェクトからしか追加できないので、
    # CSVデータを読み込んで各データをタプルで持つリストを作る
    df_tuple_list = []
    for record in data:
        converted_record = [float(value) for value in record]
        df_tuple_list.append(tuple(converted_record))

    # df_tuple_listの各要素を頂点に持つグラフGを構成する
    G = nx.Graph()
    G.add_nodes_from(df_tuple_list)

    while True:
        C = [Gc for Gc in nx.connected_components(G) if len(Gc) < k]  # 頂点の数がk個未満の連結成分を取得
        if len(C) == 0:  # 連結成分がない場合、ループを終了
            break
        for Gc in C:
            remaining_nodes = G.nodes() - Gc  # Gcを含まない頂点の集合
            # 連結成分Gcの各頂点に対して、Gc以外の点で距離が近いm個の頂点との辺を追加する
            for v1 in Gc:
                find_m_nearest_points_list = find_m_nearest_points(remaining_nodes, v1, m)
                for v2 in find_m_nearest_points_list:
                    G.add_edge(v1, tuple(v2))  # 最近傍の頂点の間に辺を追加
    return G


def find_farthest_vertex(G, start):
    distances = distance.cdist([np.array(start)], np.array(list(G))).flatten()  # ユークリッド距離を計算
    sorted_indices = np.argsort(distances)  # 距離の昇順にソートされたインデックスを取得
    farthest_indices = sorted_indices[-1]  # 距離が最大のインデックスを取得
    farthest_vertex = list(G)[farthest_indices]  # 距離が最大の点を取得
    return farthest_vertex

# 与えられた頂点集合の重心を返す関数
def calculate_centroid(points):
    points_array = np.array(points)
    num_points = points_array.shape[0]
    centroid = np.sum(points_array, axis=0) / num_points
    return centroid


def find_nearest_neighbor(points, target_point):
    distances = distance.cdist([target_point], points).flatten()  # ユークリッド距離を計算
    nearest_index = np.argsort(distances)[0]  # 最短距離のインデックスを取得
    nearest_point = points[nearest_index]  # 最短距離の点を取得
    return nearest_point


def Partition(G, k):
    result = []
    components = nx.connected_components(G)
    for component in components:
        G_component = G.subgraph(component)
        result.extend(PartitionCC(G_component, k))
    return result

# make_graph関数で作成したグラフをk個の頂点以上からなる適切なグループに分離する関数
def PartitionCC(Gc, k):
    if len(Gc) < 2 * k:
        return [list(Gc.nodes())]

    G1 = Gc.copy()
    v = random.choice(list(G1))
    farthest = find_farthest_vertex(G1, v)
    while True:
        G1.remove_node(farthest)
        C = [G1c for G1c in nx.connected_components(G1) if len(G1c) < k]
        for G1c in C:
            for c in G1c:
                G1.remove_node(c)
        # G1以外の頂点集合のサブグラフG2を作成
        sub_node = Gc.nodes() - G1
        G2 = Gc.subgraph(sub_node)
        if len(G2) >= k:
            break
        farthest = tuple(find_nearest_neighbor(list(G1), calculate_centroid(G2)))
    if len(G1) == 0:
        return [list(Gc.nodes())]
    return Partition(G1, k) + Partition(G2, k)

# 各グループに分離したデータをそのグループの(算術)平均値に変換する関数
def arithmetic_mean(data):
    result = []
    for sublist in data:
        sublist_avg = tuple(sum(pair) / len(pair) for pair in zip(*sublist))
        anonymized_data = [sublist_avg] * len(sublist)
        result.append(anonymized_data)
    return result

# Mondorian関数で分割したデータをCSVファイルから、分割した各データ単位に読み込む関数
def read_csv_rows(filename, start_row, end_row):
    rows = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader, start=1):
            if i >= start_row and i <= end_row:
                rows.append(row)
            elif i > end_row:
                break
    return rows


if __name__ == '__main__':
    data_csv = sys.argv[1]
    if len(sys.argv) > 1:
        relax = sys.argv[2]
    else:
        relax = False
    result_mondorian_csv = 'result_data/result_mondorian.csv'
    df = pd.read_csv(data_csv, skiprows=1)
    # Mondorianと友引法を併用する場合のMondorian関数で使用するkの値は320位がよい。
    # README参照
    k = 320
    df_list = df.to_numpy().tolist()
    start_time = time.time()
    result, eval_r, length_list = mondrian(df_list, k, relax)
    rtime = time.time()
    print('mondorian_time : ' + str(rtime - start_time))
    result_df = pd.DataFrame(result)
    result_df.to_csv(result_mondorian_csv, header=False, index=False)
    k = int(sys.argv[2])
    m = int(sys.argv[3])
    # 空のデータフレームを 'result.csv' に保存（上書き）
    empty_df = pd.DataFrame()
    empty_df.to_csv('result_data/result.csv', header=False, index=False)
    # 空間分割した各データに対して友引法を適用
    start_row = 1
    start_time = time.time()
    for i in range(len(length_list)):
        end_row = start_row + length_list[i] - 1
        data = read_csv_rows(result_mondorian_csv, start_row, end_row)
        lists = Partition(make_graph(data, k, m), k)
        result_mean = arithmetic_mean(lists)
        result = list(itertools.chain.from_iterable(result_mean))
        results = pd.DataFrame(result)
        results.to_csv('result_data/result.csv', mode='a', header=False, index=False)
        start_row = end_row + 1
    rtime = time.time()
    print('tomobiki_time : ' + str(rtime - start_time))
