import numpy as np
from scipy.fft import rfft, rfftfreq
from signal_generator import FS, F0, F_LINE, SIDEBAND_LOW, SIDEBAND_HIGH


# ================= 通用辅助 =================
def _peak_at(yf, xf, freq, window=2):
    if freq <= 0:
        return 0.0
    idx = np.argmin(np.abs(xf - freq))
    lo = max(0, idx - window)
    hi = min(len(yf), idx + window + 1)
    return float(np.max(yf[lo:hi]))


# ================= 时域特征 =================
def time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurtosis = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-6)
    crest_factor = peak / (rms + 1e-6)
    return {
        "RMS": float(rms),
        "峰值": float(peak),
        "峭度": float(kurtosis),
        "峰值因子": float(crest_factor),
    }


# ================= 振动频谱分析 =================
def fft_analysis(signal, fs=FS, f0=F0):
    N = len(signal)
    yf = np.abs(rfft(signal)) / N * 2
    xf = rfftfreq(N, 1/fs)

    features = {
        "1x": _peak_at(yf, xf, f0),
        "2x": _peak_at(yf, xf, 2 * f0),
        "3x": _peak_at(yf, xf, 3 * f0),
        "高频能量": float(np.mean(yf[xf > 5 * f0])) if np.any(xf > 5 * f0) else 0.0,
    }
    if len(yf) > 1:
        dominant_idx = np.argmax(yf[1:]) + 1
        features["主导频率"] = float(xf[dominant_idx])
    else:
        features["主导频率"] = 0.0
    features["1x_2x比值"] = float(features["1x"] / (features["2x"] + 1e-6))

    # 合并时域
    features.update(time_domain_features(signal))
    return features


def features_to_text(features):
    return f"""
【振动频谱分析】：
- 1x（转频）：{features['1x']:.4f}
- 2x：{features['2x']:.4f}
- 3x：{features['3x']:.4f}
- 高频能量：{features['高频能量']:.4f}
- 主导频率：{features['主导频率']:.2f} Hz
- 1x/2x比值：{features['1x_2x比值']:.2f}
- RMS：{features['RMS']:.4f}
- 峭度：{features['峭度']:.2f}（>3有冲击）
- 峰值因子：{features['峰值因子']:.2f}（>4有冲击）
"""


# ================= 电流频谱分析 =================
def current_spectrum_analysis(signal, fs=FS):
    """
    电流频谱特征：
    - 50Hz 基频
    - 2sf 边带（转子断条核心特征）
    - 3次/5次谐波（定子故障）
    - 100Hz（负序，电压不平衡）
    """
    N = len(signal)
    yf = np.abs(rfft(signal)) / N * 2
    xf = rfftfreq(N, 1/fs)

    features = {
        "50Hz基频": _peak_at(yf, xf, F_LINE),
        "边带低频": _peak_at(yf, xf, SIDEBAND_LOW),
        "边带高频": _peak_at(yf, xf, SIDEBAND_HIGH),
        "3次谐波": _peak_at(yf, xf, 3 * F_LINE),
        "5次谐波": _peak_at(yf, xf, 5 * F_LINE),
        "100Hz": _peak_at(yf, xf, 2 * F_LINE),
    }

    base = features["50Hz基频"] + 1e-6
    features["边带比"] = float((features["边带低频"] + features["边带高频"]) / base)
    features["谐波能量"] = float(features["3次谐波"] + features["5次谐波"])
    features["100Hz比"] = float(features["100Hz"] / base)

    return features


def current_features_to_text(features):
    return f"""
【电流频谱分析】：
- 50Hz基频：{features['50Hz基频']:.4f}
- 2sf边带（低频47.5Hz）：{features['边带低频']:.4f}
- 2sf边带（高频52.5Hz）：{features['边带高频']:.4f}
- 3次谐波（150Hz）：{features['3次谐波']:.4f}
- 5次谐波（250Hz）：{features['5次谐波']:.4f}
- 100Hz（负序）：{features['100Hz']:.4f}
- 边带比：{features['边带比']:.4f}（>0.1 提示转子断条）
- 谐波能量：{features['谐波能量']:.4f}（>0.5 提示定子故障）
- 100Hz比：{features['100Hz比']:.4f}（>0.1 提示电压不平衡）
"""


if __name__ == "__main__":
    from signal_generator import get_signals

    for fault in ["normal", "unbalance", "broken_bar", "stator_fault",
                  "eccentricity", "misalignment", "looseness", "voltage_unbalance"]:
        t, vib, cur = get_signals(fault)
        vf = fft_analysis(vib)
        cf = current_spectrum_analysis(cur)
        print(f"\n{'='*50}")
        print(f"故障：{fault}")
        print(f"  振动 1x={vf['1x']:.3f} 2x={vf['2x']:.3f} 峭度={vf['峭度']:.2f}")
        print(f"  电流 边带比={cf['边带比']:.4f} 谐波={cf['谐波能量']:.4f} 100Hz比={cf['100Hz比']:.4f}")