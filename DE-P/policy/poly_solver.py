import numpy as np
from scipy.spatial import distance


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """单轴五次多项式：输入初始/目标状态（位置、速度、加速度）和总时间，求解多项式系数"""
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1/2, 0, 0, 0],
            [-10/t**3, -6/t**2, -3/(2*t), 10/t**3, -4/t**2, 1/(2*t)],
            [15/t**4, 8/t**3, 3/(2*t**2), -15/t**4, 7/t**3, -1/t**2],
            [-6/t**5, -3/t**4, -1/(2*t**3), 6/t**5, -3/t**4, 1/(2*t**3)]
        ])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t ** 2

    def get_acceleration(self, t):
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t ** 2 + 20 * self.A[5] * t ** 3

    def get_velocity(self, t):
        return (self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t ** 2 + 
                4 * self.A[4] * t ** 3 + 5 * self.A[5] * t ** 4)

    def get_position(self, t):
        return (self.A[0] + self.A[1] * t + self.A[2] * t ** 2 + self.A[3] * t ** 3 + 
                self.A[4] * t ** 4 + self.A[5] * t ** 5)


class MultiAxisPoly5Solver:
    """适配三维空间的多轴五次多项式"""
    def __init__(self, uav_state, target_pos, Tf, max_vel=2.0, max_acc=2.0):
        self.Tf = Tf
        self.axes = ['x', 'y', 'z']
        self.solver_dict = {}

        # 计算目标速度
        dir_to_target = target_pos - uav_state.position
        dir_norm = np.linalg.norm(dir_to_target) + 1e-5
        target_vel = (dir_to_target / dir_norm) * min(max_vel, dir_norm / Tf)
        
        # 目标加速度
        target_acc = np.zeros(3)
        target_acc = np.clip(target_acc, -max_acc, max_acc)

        # 为各轴创建Poly5Solver
        for i, axis in enumerate(self.axes):
            pos0 = uav_state.position[i]
            vel0 = uav_state.velocity[i]
            acc0 = uav_state.acceleration[i]
            pos1 = target_pos[i]
            vel1 = target_vel[i]
            acc1 = target_acc[i]
            
            self.solver_dict[axis] = Poly5Solver(pos0, vel0, acc0, pos1, vel1, acc1, Tf)

    def get_state_at_time(self, t):
        """获取指定时间t的三维全状态（位置、速度、加速度）"""
        t = np.clip(t, 0, self.Tf)
        pos = np.array([self.solver_dict[ax].get_position(t) for ax in self.axes])
        vel = np.array([self.solver_dict[ax].get_velocity(t) for ax in self.axes])
        acc = np.array([self.solver_dict[ax].get_acceleration(t) for ax in self.axes])
        return pos, vel, acc


class Viewpoint:
    """视点类：包含位置和偏航角"""
    def __init__(self, position, yaw):
        self.position = np.array(position)  # 三维坐标 [x, y, z]
        self.yaw = yaw  # 偏航角（弧度）


class UAVState:
    """无人机状态类"""
    def __init__(self, position, velocity, acceleration, yaw):
        self.position = np.array(position)    # [x, y, z]
        self.velocity = np.array(velocity)    # [vx, vy, vz]
        self.acceleration = np.array(acceleration)  # [ax, ay, az]
        self.yaw = yaw  # 当前偏航角（弧度）


def viewpoints_in_local(vps, current_pos, sensor_range, target_vp, angle_thresh=np.pi/2):
    """筛选局部范围内的有效视点"""
    valid_vps = []
    for vp in vps:
        # 距离筛选
        dist = distance.euclidean(vp.position, current_pos)
        if dist >= sensor_range:
            continue
        
        # 可见性筛选（简化：无遮挡）
        is_visible = True
        
        # 角度筛选
        vec1 = vp.position - current_pos
        vec2 = target_vp.position - current_pos
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))
        
        if is_visible and angle < angle_thresh:
            valid_vps.append(vp)
    return valid_vps


def find_middle_yaw(valid_vps, current_yaw):
    """找到最大偏航变化的中间视点"""
    max_delta = 0
    middle_yaw = current_yaw
    for vp in valid_vps:
        delta = abs(vp.yaw - current_yaw)
        delta = min(delta, 2 * np.pi - delta)
        
        if delta > max_delta:
            max_delta = delta
            middle_yaw = vp.yaw
    return middle_yaw


def estimate_min_yaw_time(yaw1, yaw2, max_yaw_rate):
    """估算偏航角变化所需最小时间"""
    delta = abs(yaw2 - yaw1)
    delta = min(delta, 2 * np.pi - delta)
    return delta / max_yaw_rate if max_yaw_rate > 0 else np.inf


def estimate_min_flight_time(uav_state, target_pos, max_vel=2.0, max_acc=2.0):
    """估算飞行所需最小时间"""
    dist = distance.euclidean(uav_state.position, target_pos)
    if dist < 1e-5:
        return 0.0
    
    # 基于最大速度的时间
    time_vel = dist / max_vel
    
    # 基于最大加速度的时间
    time_acc_to_max_vel = max_vel / max_acc
    dist_acc = 0.5 * max_acc * time_acc_to_max_vel ** 2
    
    if dist_acc * 2 >= dist:
        time_acc = np.sqrt(2 * dist / max_acc)
    else:
        dist_const = dist - 2 * dist_acc
        time_const = dist_const / max_vel
        time_acc = 2 * time_acc_to_max_vel + time_const
    
    return max(time_vel, time_acc)


def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.3):
    """计算偏航角"""
    YAW_DOT_MAX_PER_SEC = max_yaw_rate * np.pi
    
    # 归一化速度方向（仅取水平分量）
    vel_dir_2d = vel_dir[:2]
    vel_dir_2d = vel_dir_2d / (np.linalg.norm(vel_dir_2d) + 1e-5)
    
    # 归一化目标方向
    goal_dir_2d = goal_dir[:2]
    goal_dist_2d = np.linalg.norm(goal_dir_2d)
    goal_dir_2d = goal_dir_2d / (goal_dist_2d + 1e-5)
    
    # 动态权重计算
    goal_yaw = np.arctan2(goal_dir_2d[1], goal_dir_2d[0])
    delta_yaw = goal_yaw - last_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    weight = 6 * abs(delta_yaw) / np.pi
    
    # 期望方向与临时偏航角
    dir_des_2d = vel_dir_2d + weight * goal_dir_2d
    yaw_temp = np.arctan2(dir_des_2d[1], dir_des_2d[0]) if goal_dist_2d > 0.2 else last_yaw
    
    # 限制偏航角变化量
    max_yaw_change = YAW_DOT_MAX_PER_SEC * dt
    delta_yaw_temp = yaw_temp - last_yaw
    delta_yaw_temp = (delta_yaw_temp + np.pi) % (2 * np.pi) - np.pi
    
    if delta_yaw_temp > max_yaw_change:
        yaw = last_yaw + max_yaw_change
        yawdot = YAW_DOT_MAX_PER_SEC
    elif delta_yaw_temp < -max_yaw_change:
        yaw = last_yaw - max_yaw_change
        yawdot = -YAW_DOT_MAX_PER_SEC
    else:
        yaw = yaw_temp
        yawdot = delta_yaw_temp / dt
    
    yaw = yaw % (2 * np.pi)
    return yaw, yawdot


def adaptive_full_trajectory_planning(
    vps, target_vp, uav_state, max_vel=2.0, max_acc=2.0, 
    max_yaw_rate=0.3, sensor_range=4.5, d_thr=4.0, tau=1.2, cs_threshold=0.5, num_time_steps=50
):
    """自适应全状态轨迹规划"""
    # 筛选有效视点
    valid_vps = viewpoints_in_local(vps, uav_state.position, sensor_range, target_vp)
    Nv = len(valid_vps)
    target_pos = target_vp.position
    Dk = distance.euclidean(uav_state.position, target_pos)

    # 初始化轨迹参数
    xi_m = target_vp.yaw
    Tmin_single = estimate_min_flight_time(uav_state, target_pos, max_vel, max_acc)
    Treal = max(Tmin_single, estimate_min_yaw_time(uav_state.yaw, target_vp.yaw, max_yaw_rate))

    # 判断两阶段规划
    if Nv > 1:
        xi_m = find_middle_yaw(valid_vps, uav_state.yaw)
        T1_yaw = estimate_min_yaw_time(uav_state.yaw, xi_m, max_yaw_rate)
        T2_yaw = estimate_min_yaw_time(xi_m, target_vp.yaw, max_yaw_rate)
        Tmin_two_stage = tau * (T1_yaw + T2_yaw)
        
        flight_time_est = Dk / np.linalg.norm(uav_state.velocity) if np.linalg.norm(uav_state.velocity) > 0 else np.inf
        cs_k = 0.6  # 简化处理
        
        if Tmin_two_stage <= flight_time_est or cs_k > cs_threshold:
            Tmin_flight = estimate_min_flight_time(uav_state, target_pos, max_vel, max_acc)
            Treal = max(Tmin_flight, Tmin_two_stage)

    # 生成位置/速度/加速度轨迹
    traj_solver = MultiAxisPoly5Solver(uav_state, target_pos, Treal, max_vel, max_acc)
    times = np.linspace(0, Treal, num_time_steps)
    
    # 生成偏航角轨迹
    traj_pos = []
    traj_vel = []
    traj_acc = []
    traj_yaw = []
    traj_yawdot = []
    
    last_yaw = uav_state.yaw
    dt = times[1] - times[0] if num_time_steps > 1 else 0
    
    for t in times:
        pos, vel, acc = traj_solver.get_state_at_time(t)
        traj_pos.append(pos)
        traj_vel.append(vel)
        traj_acc.append(acc)
        
        goal_dir = target_pos - pos
        yaw, yawdot = calculate_yaw(vel, goal_dir, last_yaw, dt, max_yaw_rate)
        traj_yaw.append(yaw)
        traj_yawdot.append(yawdot)
        
        last_yaw = yaw

    # 转换为numpy数组
    traj_pos = np.array(traj_pos)
    traj_vel = np.array(traj_vel)
    traj_acc = np.array(traj_acc)
    traj_yaw = np.array(traj_yaw)
    traj_yawdot = np.array(traj_yawdot)
    
    return times, traj_pos, traj_vel, traj_acc, traj_yaw, traj_yawdot


# 示例调用
if __name__ == "__main__":
    # 初始化无人机状态
    uav_init_state = UAVState(
        position=[0, 0, 1],
        velocity=[1.0, 0, 0],
        acceleration=[0, 0, 0],
        yaw=0.0
    )

    # 定义视点集合
    viewpoints = [
        Viewpoint(position=[3, 2, 1], yaw=np.pi/4),
        Viewpoint(position=[4, 1, 1], yaw=np.pi/2),
        Viewpoint(position=[3, -2, 1], yaw=3*np.pi/4),
        Viewpoint(position=[5, 0, 1], yaw=0.0)
    ]
    target_viewpoint = viewpoints[-1]

    # 执行轨迹规划
    times, traj_pos, traj_vel, traj_acc, traj_yaw, traj_yawdot = adaptive_full_trajectory_planning(
        vps=viewpoints,
        target_vp=target_viewpoint,
        uav_state=uav_init_state,
        max_vel=2.0,
        max_acc=1.0,
        max_yaw_rate=0.3,
        sensor_range=4.5,
        num_time_steps=30
    )

    # 打印结果（修复了格式化问题）
    print("="*80)
    print("自适应全状态轨迹规划结果（前5个时间步）：")
    print("="*80)
    # 打印表头
    print(f"{'时间(s)':<10} {'位置(x,y,z)':<25} {'速度(vx,vy,vz)':<25} {'偏航角(°)':<10}")
    print("-"*80)
    # 打印数据行，修复了元组格式化问题
    for i in range(min(5, len(times))):
        t = round(times[i], 2)
        # 格式化位置元组
        pos_str = f"({traj_pos[i][0]:.2f}, {traj_pos[i][1]:.2f}, {traj_pos[i][2]:.2f})"
        # 格式化速度元组
        vel_str = f"({traj_vel[i][0]:.2f}, {traj_vel[i][1]:.2f}, {traj_vel[i][2]:.2f})"
        yaw_deg = round(np.rad2deg(traj_yaw[i]), 1)
        print(f"{t:<10} {pos_str:<25} {vel_str:<25} {yaw_deg:<10}")
    print("="*80)
    print(f"轨迹总时间：{round(times[-1], 2)}秒")
    print(f"初始位置→目标位置：({traj_pos[0][0]:.2f}, {traj_pos[0][1]:.2f}, {traj_pos[0][2]:.2f}) → "
          f"({traj_pos[-1][0]:.2f}, {traj_pos[-1][1]:.2f}, {traj_pos[-1][2]:.2f})")
    print("="*80)
