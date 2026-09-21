import numpy as np

FS = 12000  # 采样率 12kHz
RPM = 1730
F0 = RPM / 60  # 转频 28.83 Hz

# ===== 电流信号参数（中国50Hz电网） =====
F_LINE = 50.0           # 电源频率
SLIP = 0.03             # 转差率
F_SLIP = SLIP * F_LINE  # 2.5 Hz
SIDEBAND_LOW = F_LINE - 2 * F_SLIP   # 47.5 Hz
SIDEBAND_HIGH = F_LINE + 2 * F_SLIP  # 52.5 Hz


# ================= 振动信号 =================
def generate_normal(t):
    return 0.05 * np.random.randn(len(t))

def generate_unbalance(t):
    return 0.5 * np.sin(2 * np.pi * F0 * t) + 0.05 * np.random.randn(len(t))

def generate_broken_bar(t):
    base = 0.3 * np.sin(2 * np.pi * F0 * t)
    sideband = 0.15 * np.sin(2 * np.pi * (F0 + 2) * t) + 0.15 * np.sin(2 * np.pi * (F0 - 2) * t)
    return base + sideband + 0.05 * np.random.randn(len(t))

def generate_eccentricity(t):
    return 0.4 * np.sin(2 * np.pi * 2 * F0 * t) + 0.1 * np.sin(2 * np.pi * F0 * t) + 0.05 * np.random.randn(len(t))

def generate_stator_fault(t):
    return 0.2 * np.sin(2 * np.pi * F0 * t) + 0.15 * np.sin(2 * np.pi * 5 * F0 * t) + 0.1 * np.random.randn(len(t))

def generate_misalignment(t):
    return 0.4 * np.sin(2 * np.pi * 2 * F0 * t) + 0.2 * np.sin(2 * np.pi * F0 * t) + 0.05 * np.random.randn(len(t))

def generate_looseness(t):
    signal = 0.1 * np.sin(2 * np.pi * F0 * t)
    for i in range(0, len(t), len(t) // 20):
        signal[i:i+20] += np.random.randn(20) * 5
    return signal + 0.05 * np.random.randn(len(t))

def generate_voltage_unbalance(t):
    return 0.3 * np.sin(2 * np.pi * 2 * F0 * t) + 0.2 * np.sin(2 * np.pi * F0 * t) + 0.08 * np.random.randn(len(t))


# ================= 电流信号 =================
def cur_normal(t):
    """正常电流：纯50Hz"""
    return 5.0 * np.sin(2 * np.pi * F_LINE * t) + 0.05 * np.random.randn(len(t))

def cur_broken_bar(t):
    """转子断条：50Hz + 2sf边带（核心特征）"""
    base = 5.0 * np.sin(2 * np.pi * F_LINE * t)
    sb_low = 0.8 * np.sin(2 * np.pi * SIDEBAND_LOW * t)
    sb_high = 0.8 * np.sin(2 * np.pi * SIDEBAND_HIGH * t)
    return base + sb_low + sb_high + 0.05 * np.random.randn(len(t))

def cur_stator_fault(t):
    """定子匝间短路：3次、5次谐波增强"""
    return (5.0 * np.sin(2 * np.pi * F_LINE * t)
            + 0.9 * np.sin(2 * np.pi * 3 * F_LINE * t)
            + 0.6 * np.sin(2 * np.pi * 5 * F_LINE * t)
            + 0.1 * np.random.randn(len(t)))

def cur_voltage_unbalance(t):
    """电压不平衡：100Hz负序分量 + 轻微幅值波动"""
    return (5.0 * np.sin(2 * np.pi * F_LINE * t)
            + 0.9 * np.sin(2 * np.pi * 2 * F_LINE * t)
            + 0.15 * np.random.randn(len(t)))

def cur_eccentricity(t):
    """气隙偏心：弱2sf边带"""
    base = 5.0 * np.sin(2 * np.pi * F_LINE * t)
    sb_low = 0.35 * np.sin(2 * np.pi * SIDEBAND_LOW * t)
    sb_high = 0.35 * np.sin(2 * np.pi * SIDEBAND_HIGH * t)
    return base + sb_low + sb_high + 0.05 * np.random.randn(len(t))


# ================= 映射表 =================
VIB_GENERATORS = {
    "normal": generate_normal,
    "unbalance": generate_unbalance,
    "broken_bar": generate_broken_bar,
    "eccentricity": generate_eccentricity,
    "stator_fault": generate_stator_fault,
    "misalignment": generate_misalignment,
    "looseness": generate_looseness,
    "voltage_unbalance": generate_voltage_unbalance,
}

CUR_GENERATORS = {
    "normal": cur_normal,
    "unbalance": cur_normal,              # 不平衡不产生电流特征
    "misalignment": cur_normal,           # 对中不良不产生电流特征
    "looseness": cur_normal,              # 松动不产生电流特征
    "eccentricity": cur_eccentricity,     # 偏心弱边带
    "broken_bar": cur_broken_bar,         # 断条强边带
    "stator_fault": cur_stator_fault,     # 定子谐波
    "voltage_unbalance": cur_voltage_unbalance,  # 电压100Hz
}


# ================= 对外接口 =================
def get_signal(fault_type, duration=1.0):
    """只返回振动信号（兼容旧版调用）"""
    t = np.linspace(0, duration, int(FS * duration))
    if fault_type not in VIB_GENERATORS:
        raise ValueError(f"未知故障类型: {fault_type}")
    return t, VIB_GENERATORS[fault_type](t)


def get_signals(fault_type, duration=1.0):
    """同时返回振动 + 电流信号（新版）"""
    t = np.linspace(0, duration, int(FS * duration))
    if fault_type not in VIB_GENERATORS:
        raise ValueError(f"未知故障类型: {fault_type}")
    vib = VIB_GENERATORS[fault_type](t)
    cur = CUR_GENERATORS[fault_type](t)
    return t, vib, cur


if __name__ == "__main__":
    for fault in VIB_GENERATORS:
        t, vib, cur = get_signals(fault)
        print(f"{fault:20s} 振动幅值=[{vib.min():.2f},{vib.max():.2f}]  电流幅值=[{cur.min():.2f},{cur.max():.2f}]")