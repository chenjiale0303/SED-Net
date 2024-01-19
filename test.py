
import sys
import logging
import json
import os

from read_config import Config
config = Config(sys.argv[1])
GPU = config.gpu

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

from shutil import copyfile
import numpy as np


from gen_test_vis import COLORS_TYPE, visual_labels

from src.dataset_segments import ori_simple_data

from src.smooth_normal_matrix import hpnet_process

# COLORS_TYPE = np.array([[0, 210, 93], [1, 206, 57], [3, 45, 16], [5, 175, 94], [9, 112, 237], [9, 74, 234], [10, 96, 103], [13, 63, 159], [14, 94, 63], [15, 249, 175], [23, 88, 151], [26, 4, 3], [26, 128, 106], [30, 23, 96], [33, 18, 148], [36, 230, 28], [36, 140, 175], [38, 84, 73], [43, 71, 97], [46, 198, 113], [53, 192, 208], [55, 53, 248], [56, 139, 217], [61, 157, 87], [61, 94, 172], [61, 90, 137], [66, 14, 155], [67, 147, 71], [67, 101, 215], [72, 223, 16], [73, 241, 192], [76, 133, 147], [79, 92, 48], [79, 179, 141], [81, 1, 52], [87, 132, 247], [87, 216, 17], [88, 68, 118], [88, 243, 78], [90, 250, 203], [93, 172, 60], [95, 17, 201], [96, 159, 73], [98, 244, 99], [99, 46, 125], [100, 195, 53], [105, 236, 84], [110, 141, 224], [111, 117, 97], [112, 143, 106], [116, 105, 90], [123, 37, 147], [125, 101, 19], [127, 222, 80], [134, 55, 230], [134, 78, 68], [138, 246, 23], [142, 247, 156], [144, 4, 193], [149, 37, 236], [153, 20, 17], [159, 36, 141], [167, 105, 29], [168, 156, 246], [168, 7, 167], [171, 22, 157], [172, 137, 36], [172, 10, 42], [172, 210, 64], [172, 237, 165], [179, 51, 81], [184, 135, 238], [185, 99, 150], [201, 112, 162], [203, 141, 165], [207, 246, 231], [212, 201, 183], [213, 194, 221], [213, 9, 225], [222, 40, 202], [222, 141, 224], [222, 120, 141], [222, 59, 193], [223, 143, 128], [224, 207, 235], [225, 38, 199], [226, 15, 187], [227, 245, 72], [228, 131, 251], [232, 77, 132], [237, 179, 34], [237, 153, 53], [237, 91, 36], [239, 111, 214], [246, 137, 195], [249, 228, 168], [252, 65, 95], [254, 1, 39], [255, 136, 104], [255, 61, 23]])

COLORS_TYPE = np.array([[0, 41, 108], [4, 202, 234], [5, 87, 101], [6, 185, 227], [7, 189, 23], [9, 184, 244], [9, 202, 109], [10, 75, 73], [12, 220, 64], [16, 69, 67], [16, 30, 85], [22, 134, 101], [26, 248, 184], [26, 20, 158], [27, 249, 99], [28, 72, 170], [29, 85, 49], [32, 65, 31], [36, 101, 155], [37, 203, 49], [39, 152, 127], [39, 128, 251], [41, 17, 21], [43, 40, 81], [44, 201, 0], [45, 110, 12], [49, 104, 99], [62, 45, 157], [63, 16, 234], [70, 11, 13], [76, 197, 157], [79, 49, 171], [81, 244, 171], [81, 189, 144], [82, 238, 65], [84, 221, 141], [85, 255, 213], [93, 96, 217], [101, 116, 225], [108, 97, 122], [109, 11, 177], [114, 204, 107], [123, 110, 212], [123, 211, 7], [128, 138, 243], [128, 180, 225], [131, 4, 105], [132, 49, 65], [134, 210, 26], [143, 24, 207], [147, 179, 123], [147, 73, 115], [149, 189, 238], [153, 81, 31], [164, 28, 94], [166, 18, 199], [169, 17, 41], [170, 34, 109], [172, 209, 32], [174, 168, 122], [174, 188, 224], [178, 185, 3], [179, 128, 1], [181, 79, 234], [183, 64, 0], [184, 212, 169], [187, 133, 130], [189, 125, 81], [192, 154, 233], [198, 222, 132], [199, 54, 100], [199, 111, 103], [201, 6, 215], [203, 89, 113], [204, 23, 37], [205, 152, 174], [205, 126, 210], [206, 89, 198], [208, 176, 187], [214, 111, 243], [217, 55, 227], [217, 99, 221], [218, 70, 81], [220, 234, 158], [221, 63, 15], [222, 114, 213], [224, 19, 168], [227, 21, 40], [228, 81, 40], [232, 132, 165], [233, 139, 61], [236, 136, 241], [237, 60, 61], [238, 166, 211], [238, 67, 66], [240, 237, 126], [242, 141, 39], [242, 227, 107], [251, 121, 4], [252, 146, 95]])

def guard_mean_shift(ms, embedding, quantile, iterations, kernel_type="gaussian"):

        while True:
            _, center, bandwidth, cluster_ids = ms.mean_shift(
                embedding, 10000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 1.2
            else:
                break
        return center, bandwidth, cluster_ids


program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")

# from src.my_pointnext import PointNeXt_S_Seg

import torch
from src.SEDNet import SEDNet



from src.segment_loss import EmbeddingLoss
from src.segment_utils import SIOU_matched_segments_usecd, compute_type_miou_abc
from src.segment_utils import to_one_hot, SIOU_matched_segments

from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments

# test configs
HPNet_embed = True # ========================= default True 
NORMAL_SMOOTH_W = 0.5  # =================== default 0.5
Concat_TYPE_C6 = False # ====================== default False
Concat_EDGE_C2 = False # ====================== default False
INPUT_SIZE = 10000 # =====input pc num, default 10000
my_knn = 64 # ==== default 64
use_hpnet_type_iou = False
drop_out_num = 2000 # ====== type seg rand drop  


prefix="/data/ytliu/parsenet/" # test dataset path prefix
starts = 0  # default 0 

if HPNet_embed:
    print("uisng HPNet embeding way!!!!")



SAVE_VIZ = not sys.argv[2] == "NoSave"

# type 结果进行数据增强投票
MULTI_VOTE = sys.argv[3] == "multi_vote"
if MULTI_VOTE:
    print("type_multi_vote")


# type 结果进行数据增强投票
fold5Drop = sys.argv[4] == "fold5drop"
if fold5Drop:
    print("type_fold5drop")  # ======= 效果好


if_normals = config.normals

Use_MyData = True if config.dataset == "my" else False
# =============== test dataset config

if Use_MyData:
    config.num_val = config.num_test = 2700
else:
    config.num_val = config.num_test = 4163


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

fn = "TEST_SEDNet_{}_{}_{}{}{}{}{}{}_smW{}_in{}_knn{}{}_dropnum{}_Recall".format(
        config.pretrain_model_path.split("/")[-1], 
        "MyData" if Use_MyData else "", 
        "normal0" if config.mode==4 else "", 
        "_multi_vote_2" if MULTI_VOTE else "",
        "_fold5Drop" if fold5Drop else "",
        "_HPNet_embed" if HPNet_embed else "",
        "_ConcatType" if Concat_TYPE_C6 else "",
        "_concatEdge" if Concat_EDGE_C2 else "",
        str(NORMAL_SMOOTH_W),
        str(INPUT_SIZE),
        str(my_knn),
        "_hpnetTypeIoU" if use_hpnet_type_iou else "",
        drop_out_num
    )

file_handler = logging.FileHandler(
    f"./predictions/logs/{fn}.log", mode="a"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "./predictions/config/cfg_{}.json".format(fn), "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "./predictions/config/code_{}_{}".format(
    fn, __file__.split("/")[-1]
)
copyfile(source_file, destination_file)

userspace = ""
Loss = EmbeddingLoss(margin=1.0)

model = SEDNet(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=6,
        loss_function=Loss.triplet_loss,
        mode=5,
        num_channels=6,
        combine_label_prim=True,   # early fusion
        edge_module=True,  # add edge cls module
        late_fusion=True,  
        nn_nb=my_knn  # default is 64
    )
model_inst = SEDNet(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=6,
        loss_function=Loss.triplet_loss,
        mode=5,
        num_channels=6,
        combine_label_prim=True,   # early fusion
        edge_module=True,  # add edge cls module
        late_fusion=True,    # ======================================
        nn_nb=my_knn  # default is 64
    )

print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.cuda( )
model_inst = model_inst.cuda( )

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}
ms = MeanShift()


mix_test_dataset = ori_simple_data(if_normals=if_normals, if_train=False, starts=starts)

loader_test = torch.utils.data.DataLoader(
    mix_test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False
)

if SAVE_VIZ:
    os.makedirs(userspace + "./predictions/results/{}{}/results/".format("MyData_" if Use_MyData else "", config.pretrain_model_path), exist_ok=True)

model.eval()
model_inst.eval()

iterations = 50
quantile = 0.015

state_dict = torch.load(config.pretrain_model_path)
state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
model.load_state_dict(state_dict)


state_dict = torch.load(config.pretrain_model_type_path)
state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
model_inst.load_state_dict(state_dict)


test_res = []
test_s_iou = []
test_p_iou = []
test_g_res = []
test_s_res = []
PredictedLabels = []
PredictedPrims = []
test_p_iou_HPNET = []
test_s_recall = []

save_gt = False

# Construct a list with four item [points, labels, normals, primitives]
import open3d as o3d
gsp_test_data = [] if not os.path.exists("./gsp/gsp_test_data.pth") else torch.load("./gsp/gsp_test_data.pth")

GSP_DIR = sys.argv[5]
OUTPUT_DIR = "D:/Dataset/GSP/baselines/SED_prediction_0116"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if len(gsp_test_data) == 0 or True:
    gsp_test_data = []
    data = []
    for file in os.listdir(GSP_DIR):
        if not file.endswith(".ply"):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(GSP_DIR, file))
        points = torch.from_numpy(np.array(pcd.points, dtype=np.float32)).unsqueeze(0)
        normals = torch.from_numpy(np.array(pcd.normals, dtype=np.float32)).unsqueeze(0)
        longest_axis = (points.max(axis=1)[0] - points.min(axis=1)[0]).max(axis=1)[0]
        scale = 1 / longest_axis.max()
        # diagonal_length = torch.sqrt(torch.sum((points.max(axis=1)[0] - points.min(axis=1)[0]) ** 2, dim=1))
        points = points * scale
        normals /= torch.norm(normals, dim=2, keepdim=True)
        data.append(points)
        data.append(torch.zeros((points.shape[0], 1, 10000)))
        data.append(normals)
        data.append(torch.zeros((points.shape[0], 1, 10000)))
        data.append(file)
        data.append(scale)
        gsp_test_data.append(data)
        data = []

    torch.save(gsp_test_data, "./gsp/gsp_test_data.pth")


# for val_b_id, data in enumerate(loader_test):
for val_b_id, data in enumerate(gsp_test_data):
    points_, labels, normals_, primitives_, filename, scale = data[:6]
    
    points = points_.cuda()
    normals = normals_.cuda()
    labels = labels.numpy()
    primitives_ = primitives_.numpy()
    
    with torch.no_grad():
        if if_normals:
            _input = torch.cat([points, normals], 2)
            primitives_log_prob = model(
                _input.permute(0, 2, 1), None, False
            )[1]
            embedding, _, _, edges_pred = model_inst(
                _input.permute(0, 2, 1), None, False
            )           
        else:
            primitives_log_prob = model(
                points.permute(0, 2, 1), None, False
            )[1]
            embedding, _, _, edges_pred = model_inst(
                points.permute(0, 2, 1), None, False
            )

        if MULTI_VOTE and not fold5Drop:
            points_big = points * 1.15
            if if_normals:
                input = torch.cat([points_big, normals], 2)
                embedding_big, primitives_log_prob_big = model(
                    input.permute(0, 2, 1), None, False
                )[:2]
            else:
                embedding_big, primitives_log_prob_big = model(
                    points_big.permute(0, 2, 1), None, False
                )[:2]

            points_small = points * 0.85
            if if_normals:
                input = torch.cat([points_small, normals], 2)
                embedding_small, primitives_log_prob_small = model(
                    input.permute(0, 2, 1), None, False
                )[:2]
            else:
                embedding_small, primitives_log_prob_small = model(
                    points_small.permute(0, 2, 1), None, False
                )[:2]              

            primitives_log_prob = (primitives_log_prob + primitives_log_prob_big + primitives_log_prob_small) / 3


        if fold5Drop and not MULTI_VOTE:
            # batch_points = None
            # batch_normals = None
            total_type_pred = torch.zeros_like(primitives_log_prob).flatten()
            primitives_log_prob_batch = None
            iter_times = 10000 // drop_out_num
            for i in range(iter_times):
                index = torch.ones(points.shape, dtype=torch.bool).cuda()
                index[:, i*drop_out_num:(i+1)*drop_out_num, :] = False
                points_drop = points[index].reshape((1, 10000 - drop_out_num, 3))
                normals_drop = normals[index].reshape((1, 10000 - drop_out_num, 3))
                batch_points = points_drop
                batch_normals = normals_drop
                
                if primitives_log_prob_batch is None:
                    if if_normals:
                        input = torch.cat([batch_points, batch_normals], 2)
                        primitives_log_prob_batch = model(
                                input.permute(0, 2, 1), None, False
                            )[1]
                    else:
                        primitives_log_prob_batch = model(
                                batch_points.permute(0, 2, 1), None, False
                            )[1]  
                else:
                    if if_normals:
                        input = torch.cat([batch_points, batch_normals], 2)
                        primitives_log_prob_batch = torch.cat([primitives_log_prob_batch, model(
                                input.permute(0, 2, 1), None, False
                            )[1]], dim=0)
                    else:
                        primitives_log_prob_batch = torch.cat([primitives_log_prob_batch, model(
                                batch_points.permute(0, 2, 1), None, False
                            )[1]], dim=0)                     

            for i in range(iter_times):
                index = torch.ones(primitives_log_prob.shape, dtype=torch.bool).cuda()
                index[:, :, i*drop_out_num:(i+1)*drop_out_num] = False    
                total_type_pred[index.flatten()] += primitives_log_prob_batch[i].flatten()

            primitives_log_prob += total_type_pred.reshape(primitives_log_prob.shape)


        if fold5Drop and MULTI_VOTE:
            """
            data augmentation

            """
            angles = [
                torch.from_numpy(np.array(      
                     [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)).cuda( ).unsqueeze(0), 
                torch.from_numpy(np.array(      
                     [[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]], dtype=np.float32)).cuda( ).unsqueeze(0),                 
            ] 

            primitives_prob_total = None
            for R in angles:
                normals_cur = torch.bmm(normals, R)
                points_cur = torch.bmm(points, R)

                if if_normals:
                    input = torch.cat([points_cur, normals_cur], 2)
                    primitives_log_prob_cur= model(input.permute(0, 2, 1), None, False)[1]
                else:
                    primitives_log_prob_cur= model(points_cur.permute(0, 2, 1), None, False)[1]                

                total_type_pred = torch.zeros_like(primitives_log_prob_cur).flatten()  # 6 x 1w
                for i in range(5):
                    index = torch.ones(points.shape, dtype=torch.bool, device=torch.device("cuda"))
                    index[:, i*2000:(i+1)*2000, :] = False
                    points_drop = points_cur[index].reshape((1, 8000, 3))
                    normals_drop = normals_cur[index].reshape((1, 8000, 3))  # =========

                
                    if if_normals:
                        _input = torch.cat([points_drop, normals_drop], 2)
                        primitives_log_prob_batch = model(
                                _input.permute(0, 2, 1), None, False
                            )[1]
                    else:
                        primitives_log_prob_batch = model(
                                batch_points.permute(0, 2, 1), None, False
                            )[1] 
                    index = torch.ones(primitives_log_prob_cur.shape, dtype=torch.bool, device=torch.device("cuda"))
                    index[:, :, i*2000:(i+1)*2000] = False    
                    total_type_pred[index.flatten() ] += primitives_log_prob_batch.flatten()

                primitives_log_prob_cur += total_type_pred.reshape(primitives_log_prob.shape)     

                if primitives_prob_total is None:
                    primitives_prob_total =  primitives_log_prob_cur
                else:
                    primitives_prob_total += primitives_log_prob_cur

            primitives_log_prob = primitives_prob_total     


    pred_primitives = torch.max(primitives_log_prob[0], 0)[1].data.cpu().numpy()

    primitives_prob_total = None
    index = None
    total_type_pred = None
  
    if HPNet_embed:
        embedding = hpnet_process(embedding.transpose(1, 2), points, normals, id=val_b_id, 
            types=primitives_log_prob.transpose(1, 2) if Concat_TYPE_C6 else None,
            edges=edges_pred.transpose(1, 2) if Concat_EDGE_C2 else None,
            normal_smooth_w=NORMAL_SMOOTH_W, CHUNK=1000
        )
        embedding = torch.nn.functional.normalize(embedding[0], p=2, dim=1)

    else:
        embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

    _, _, cluster_ids = guard_mean_shift(
            ms, embedding, quantile, iterations, kernel_type="gaussian"
        )
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[0])
    cluster_ids = cluster_ids.data.cpu().numpy()

    def save_xyzc(filename, points, cluster_ids):
        assert points.shape[0] == cluster_ids.shape[0]
        data = np.column_stack((points, cluster_ids))
        if filename.endswith('.ply'):
            filename = filename.replace('.ply', '.xyzc')
        assert filename.endswith('.xyzc')
        np.savetxt(filename, data, fmt='%.6f %.6f %.6f %d', delimiter=' ')

    def export_surfaces(points, normals, cluster_ids, filename):
        # 将 points 和 normals 转为 numpy 数组
        points = points.numpy()
        normals = normals.numpy()
        cluster_colors = COLORS_TYPE[cluster_ids] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points / scale)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        save_xyzc(os.path.join(OUTPUT_DIR, filename), points / scale, cluster_ids)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, filename), pcd, write_ascii=False)

    # points_ shape: (1, 10000, 3)
    export_surfaces(points_[0], normals_[0], cluster_ids, filename)