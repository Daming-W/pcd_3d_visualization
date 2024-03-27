from IPython import embed
import pickle
import json
import numpy as np
import os
from tqdm import tqdm
import copy
import glob
import open3d as o3d
import pandas as pd


# def get_pcd_path(data):
    # if "lidar" in data:
        # lidar_dict = {}
        # for item in data["lidar"]:
            # lidar_dict[item["name"]] = item
            # if "center_128_lidar_scan_data" in lidar_dict:
                # break
        # pcd_path = lidar_dict["center_128_lidar_scan_data"]["oss_path_pcd_txt"]
    # elif "point_cloud_Url" in data:
        # pcd_path = data["point_cloud_Url"]
    # else:
        # assert False
    # return pcd_path


def get_lidar_pcd(file_path):
    o3d_points = pd.read_csv(file_path, delimiter=' ', skiprows=11, dtype=np.float32, names=[
                'x', 'y', 'z', 'i', 'c'], usecols=[0,1,2,3,4]).values
                # 'x', 'y', 'z', 'i', 'c', 't'], usecols=[0,1,2,3,4,5]).values
    points, intensity = o3d_points[:,
                                       0:3], o3d_points[:, 3] / 255.
    return points


# def get_pcd(file_path):
    # pcd = pypcd.point_cloud_from_path(file_path)
    # x = pcd.pc_data['x'].reshape(-1, 1)
    # y = pcd.pc_data['y'].reshape(-1, 1)
    # z = pcd.pc_data['z'].reshape(-1, 1)
    # intensity = pcd.pc_data['intensity'].reshape(-1, 1) / 255.
    # data = np.hstack((x, y, z, intensity))
    # return data


def parse_obj_get_box(obj):
    """
    haomo's obj
    """
    print(obj)
    length, width, height = obj["dimension"]["length"], obj["dimension"]["width"], obj["dimension"]["height"]
    cx, cy, cz = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]
    rot_y = obj["rotation"]["yaw"]
    return (cx, cy, cz, length, width, height, rot_y)


def parse_json_haomo(box_path):
    ret = []
    data = json.loads(open(box_path).read())
    objs = data["objects"]
    for obj in objs:
        box = parse_obj_get_box(obj)
        ret.append(box)
    return np.array(ret).reshape(-1, 7)


def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x/2.0, dim_y/2.0, dim_z/2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    # 这个时候corners还只是平行于坐标轴且以坐标原点为中心来算的.
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    # 然后根据pose,算出真实的,即RX+T
    corners = (transform_matrix[:3, :3] @
               corners.T + transform_matrix[:3, [3]]).T
    return corners


def gen_o3d_box3d_lines(objects, label_names, colors, scale=1.0, mode="center"):
    """

    """
    assert mode in ["center", "corner"]
    box3d_lines = []
    box3d_dirs = []
    if objects is None:
        return box3d_lines
    for index, obj in enumerate(objects):
        # compute corners
        if mode == "corner":
            corners = obj / scale  # 8, 3
        else:
            corners = center_box_to_corners(obj) / scale
        lines = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 5],
                [1, 4]
        ]
        if colors is not None:
            color = [colors[index] for i in range(len(lines))]
        else:
            color = [[255, 0, 0] for i in range(len(lines))]  # r g b
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(color)
        box3d_lines.append(line_set)
    return box3d_lines


def run(pcd_path, json_path, window_name):
    """
    pcd_path: 点云的路径.
    box_path: 
    里面的shape是(m, n, 8, 3) 的shape
    m代表目标的个数, n代表这个目标的框数.
    """
    # 读取点云路径.
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # points = np.fromfile(pcd_path, "float32").reshape(-1, 4)[:, :3]
    # points = np.asarray(pcd.points).reshape(-1, 3)

    # points = get_pcd(pcd_path)
    points = get_lidar_pcd(pcd_path)
    points = points[points[:, 0] >-1000.]
    points = points[points[:, 0] <1000.]
    points = points[points[:, 1] <1000.]
    points = points[points[:, 1] >-1000.]
    points = points[points[:, 2] >-20.]
    points = points[points[:, 2] <20.]
    # from IPython import embed;embed() 
    #  创建画布进行在点云上面开始画东西.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.clear_geometries()
    o3d_point_cloud = o3d.geometry.PointCloud()
    intensity = None
    label = None
    scale = 8
    vis.create_window(window_name)
    vis.get_render_option().background_color = np.asarray(
        [0.0, 0.0, 0.0])  # white background
    vis.get_render_option().point_size = 0.2
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points / scale)
    o3d_point_cloud = o3d_point_cloud.paint_uniform_color([1, 0.206, 0])
    vis.add_geometry(o3d_point_cloud)

    # 添加坐标轴
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=8/scale, origin=(0, 0, 0))
    vis.add_geometry(FOR1)

    json_data = json.loads(open(json_path).read())
    # objects = json_data["Content"]["det3d"]["result"][0]["objects"]
    objects = json_data["objects"]
    total_boxes = [parse_obj_get_box(obj) for obj in objects]
    length = len(total_boxes)
    label_names = [0] * length
    colors = [[0, 0, 255] for _ in label_names]
    line_sets = gen_o3d_box3d_lines(
        total_boxes, label_names, colors, scale=scale, mode="center")
    for line_set in line_sets:
        vis.add_geometry(line_set, False)

    vis.run()


# pcd_paths = glob.glob("/Users/pengkun/wuliu_pcd/test/pcd_128/*.pcd")
# pcd_path = "./1679538132202707.pcd"
# json_path = "./1679538132202707.json"

json_paths = glob.glob("./vis/*.json")
start = 1
# for pcd_path in pcd_paths:
for json_path in json_paths:
    print(json_path)
    pcd_path = json_path.replace(".json", ".pcd")
    # json_data = json.loads(open(json_path).read())
    # pcd_path = os.path.basename(json_data["point_cloud_Url"])
    # pcd_path = os.path.join("/Users/pengkun/wuliu_pcd/test/pcd_128", pcd_path)
    # if not os.path.exists(pcd_path):
        # continue
    # json_path = os.path.basename(pcd_path).replace(".pcd", ".json")
    # json_path = os.path.join("./test", json_path)
    run(pcd_path, json_path, str(json_path))
