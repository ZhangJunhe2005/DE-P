import os, sys
from collections import OrderedDict
import cv2
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


DATASET_IMPLEMENTATION_VERSION = "dep_static_lazy_v1"


def seed_dataset_worker(worker_id):
    info = get_worker_info()
    dataset = info.dataset
    global_seed = getattr(
        dataset, "global_seed",
        getattr(getattr(dataset, "training_config", None), "noise_seed", 0),
    )
    epoch = getattr(dataset, "epoch", getattr(dataset, "_curriculum_epoch", 0))
    seed = int(global_seed + epoch * 100003 + worker_id)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


class DEPDataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1, cache_size=128,
                 global_seed=0, data_root=None):
        super(DEPDataset, self).__init__()
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        # ramdom state: x-direction: log-normal distribution, yz-direction: normal distribution
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        self.vx_lognorm_mean = np.log(1 - cfg["vx_mean_unit"])
        self.vx_logmorm_sigma = np.log(cfg["vx_std_unit"])
        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]])
        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]])
        self.goal_length = cfg['goal_length']
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.global_seed = int(global_seed)
        self.epoch = 0
        self.implementation_version = DATASET_IMPLEMENTATION_VERSION
        self._depth_cache = OrderedDict()
        if mode == 'train': self.print_data()

        # dataset
        print("Loading", mode, "dataset, it may take a while...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = (os.path.abspath(os.path.expanduser(str(data_root))) if data_root
                    else os.path.join(base_dir, "../", cfg["dataset_path"]))
        self.image_paths, self.map_idx = [], []
        self.positions = np.empty((0, 3), dtype=np.float32)
        self.quaternions = np.empty((0, 4), dtype=np.float32)

        datafolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        datafolders.sort(key=lambda x: int(os.path.basename(x)))
        print("Datafolders:")
        for folder in datafolders:
            print("    ", folder)

        for data_idx in range(len(datafolders)):
            datafolder = datafolders[data_idx]

            image_file_names = [filename
                                for filename in os.listdir(datafolder)
                                if os.path.splitext(filename)[1] == '.png']
            image_file_names.sort(key=lambda x: int(x.split('.')[0].split("_")[1]))  # sort by filename to align with the label

            states = np.loadtxt(data_dir + f"/pose-{data_idx}.csv", delimiter=',', skiprows=1).astype(np.float32)
            positions = states[:, 0:3]
            quaternions = states[:, 3:7]

            file_names_train, file_names_val, positions_train, positions_val, quaternions_train, quaternions_val = train_test_split(
                image_file_names, positions, quaternions, test_size=val_ratio, random_state=0)

            if mode == 'train':
                selected_names = file_names_train
                self.positions = np.vstack((self.positions, positions_train.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_train.astype(np.float32)))
            elif mode == 'valid':
                selected_names = file_names_val
                self.positions = np.vstack((self.positions, positions_val.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_val.astype(np.float32)))
            else:
                raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

            self.image_paths.extend([os.path.join(datafolder, name) for name in selected_names])
            self.map_idx.extend([data_idx] * len(selected_names))
        # Historical callers may inspect img_list; it now intentionally stores
        # paths rather than decoded float32 arrays.
        self.img_list = self.image_paths

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.image_paths):<3} |  Lazy: True")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")
        print(mode.capitalize(), "data loaded!")

    def __len__(self):
        return len(self.image_paths)

    @property
    def cached_depth_count(self):
        return len(self._depth_cache)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_depth(self, path):
        key = str(path)
        if key in self._depth_cache:
            depth = self._depth_cache.pop(key)
            self._depth_cache[key] = depth
            return depth.copy()
        raw = cv2.imread(key, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"failed to read static depth: {key}")
        depth = np.expand_dims(
            cv2.resize(raw.astype(np.float32), (self.width, self.height),
                       interpolation=cv2.INTER_NEAREST) / 65535.0,
            axis=0,
        ).astype(np.float32, copy=False)
        if not np.isfinite(depth).all():
            raise ValueError(f"static depth contains NaN/Inf: {key}")
        if self.cache_size:
            self._depth_cache[key] = depth.copy()
            while len(self._depth_cache) > self.cache_size:
                self._depth_cache.popitem(last=False)
        return depth

    def __getitem__(self, item):
        depth = self._load_depth(self.image_paths[item])
        vel_b, acc_b = self._get_random_state()

        # generate random goal in front of the quadrotor.
        q_wxyz = self.quaternions[item, :]  # q: wxyz
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)  # [yaw(z) pitch(y) roll(x)]
        R_wB = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False)
        goal_w = self._get_random_goal()
        goal_b = R_wB.inv().apply(goal_w)

        random_obs = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)
        rot_wb = R_WB.as_matrix().astype(np.float32)  # transform to rot_matrix in numpy is faster than using quat in pytorch
        # vel & acc & goal are in body frame, NWU, and no-normalization
        return depth, self.positions[item], rot_wb, random_obs, self.map_idx[item]

    def _get_random_state(self):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:  # avoid outliers
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self):
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir
        return self.goal_length * goal_w_dir

    def print_data(self):
        import scipy.stats as stats
        # 计算Vx 5% ~ 95% 区间
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max

        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        import matplotlib.pyplot as plt
        # ===== 采样 =====
        N = 10000
        goals = np.array([self._get_random_goal() for _ in range(N)])
        states = np.array([self._get_random_state() for _ in range(N)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])

        x, y, z = goals[:, 0], goals[:, 1], goals[:, 2]
        yaw = np.degrees(np.arctan2(y, x))  # 水平角 [-180, 180]
        pitch = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))  # 垂直角 [-90, 90]

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

        # Goal方向角分布
        axs[0, 0].hist(yaw, bins=180)
        axs[0, 0].set_title("Goal Yaw Distribution")
        axs[0, 0].set_xlabel("Yaw (deg)")
        axs[0, 0].set_xlim([-60, 60])
        axs[0, 0].grid(True)

        axs[0, 1].hist(pitch, bins=90)
        axs[0, 1].set_title("Goal Pitch Distribution")
        axs[0, 1].set_xlabel("Pitch (deg)")
        axs[0, 1].set_xlim([-60, 60])
        axs[0, 1].grid(True)

        # Goal往图像投影分布(未考虑机体旋转)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Distribution in Image")
        axs[0, 2].set_xlabel("Yaw (deg)")
        axs[0, 2].set_ylabel("Pitch (deg)")
        axs[0, 2].set_xlim([-45, 45])
        axs[0, 2].set_ylim([-30, 30])
        axs[0, 2].grid(True)

        # Velocity分布
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)

        # Acceleration分布
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset = DEPDataset()
    dataset.plot_sample_distribution()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    start = time.time()
    for epoch in range(1):
        last = time.time()
        for i, (depth, pos, quat, obs, id) in enumerate(data_loader):
            pass
    end = time.time()

    print("加载1个epoch总耗时：", end - start)
