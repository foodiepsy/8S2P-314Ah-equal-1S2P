# -*- coding: utf-8 -*-
"""
314Ah LFP 8S2P 电池包 —— 等效 1S2P 降阶模型仿真
================================================================
参考项目 battery_314ah_8s2p (PyBaMM 电化学 + liionpack 网表求解),
本工程用「等效电路模型 (ECM)」把 8S2P 近似等效为 1S2P:

  * 真实拓扑: 8 串联 × 2 并联 (16 颗 314Ah LFP 单体)
  * 等效拓扑: 1 个「8 倍电压的大 314Ah 个体」 × 2 并联
      - 每一支路 = 把 8 颗串联单体等效成 1 个元件:
          电压 = 8 × 单体电压  (标称 25.6 V, 范围 20.0 ~ 29.2 V)
          容量 = 314 Ah        (串联不增加容量)
          内阻 R0 = 10 mΩ      (模块级: 单体 0.3mΩ×8 + 汇流排/熔断器/接触器)
          RC 极化时间常数 tau ≈ 380 s
          rc (极化电阻 R1) 未知 → 用经验值放缩等效 8S (见 EQUIV_CONFIG)

  * 工况 (与参考一致): 支路1 先单独放电 stagger_h 小时, 之后支路2 并入并联。
      阶段1 (t < T1): 仅支路1 独担整包负载电流 I_pack = c_rate × n_p × Q
      阶段2 (t≥T1): 支路2 并入, 每支路电流 = I_pack / n_p = c_rate × Q

  * 模型: 每支路一阶 Thevenin (OCV - R0 - (R1||C1)), 并联回路环流
      I_circ = (E1 - E2) / (R1_eq + R2_eq + 2·Rc),  E_b = OCV_b - V_rc_b
      支路电流 I1 = I_nom + I_circ,  I2 = I_nom - I_circ

  * 依赖: numpy, matplotlib, pandas (运行时无需 PyBaMM)。
    OCV-SOC 曲线从参考项目的 PyBaMM/Prada2013 LFP 参数一次性抽取后内嵌 (×8)。

作者: WorkBuddy   日期: 2026-07-23
"""
import os
import sys
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================================================================
# 0. 中文字体 (与参考项目一致: SimHei / 微软雅黑)
# ================================================================
_CHINESE_FONT_PATH = ""
for _cand in (
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
):
    if os.path.exists(_cand):
        _CHINESE_FONT_PATH = _cand
        break
if _CHINESE_FONT_PATH:
    try:
        fm.fontManager.addfont(_CHINESE_FONT_PATH)
        _cn_name = fm.FontProperties(fname=_CHINESE_FONT_PATH).get_name()
        plt.rcParams["font.sans-serif"] = [_cn_name, "SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    except Exception as _e:
        print(f"  [警告] 中文字体加载失败: {_e}")
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ================================================================
# 1. 等效 8S 元件配置 (EQUIV_CONFIG)
# ================================================================
# 单体 (314Ah LFP) OCV-SOC 查算表 —— 从参考项目 PyBaMM/Prada2013 准平衡
# SPM 抽取 (变量 "Bulk open-circuit voltage [V]"), 21 点。等效 8S 时 ×8。
OCV_CELL_SOC = np.array([
    0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
    0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000,
])
OCV_CELL_V = np.array([
    2.52514, 2.80417, 2.98557, 3.11399, 3.16982, 3.18654, 3.20711, 3.23362,
    3.25318, 3.26242, 3.26614, 3.26782, 3.26886, 3.27003, 3.27440, 3.29355,
    3.30997, 3.31324, 3.31421, 3.31671, 3.65000,
])


def ocv_cell(soc: np.ndarray) -> np.ndarray:
    """单体开路电压 (V), 由 SOC 查表线性插值。"""
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    return np.interp(s, OCV_CELL_SOC, OCV_CELL_V)


def ocv_equiv(soc: np.ndarray) -> np.ndarray:
    """等效 8S 元件开路电压 (V) = 8 × 单体 OCV。"""
    return 8.0 * ocv_cell(soc)


# ---- 等效 8S 元件电气参数 ----
class EquivConfig:
    """等效 8S 支路元件 + 整包配置。"""
    # 串联数 (被「压缩」为 1 个等效元件) / 并联支路数
    series_count = 8          # 8S -> 等效为 1 个 8×电压元件
    parallel_count = 2        # 2 条并联支路

    # 等效元件容量: 串联不增加容量 -> 314 Ah
    cell_capacity_ah = 314.0
    # 等效元件标称电压 = 8 × 3.2 = 25.6 V
    nominal_voltage_v = 3.2 * series_count
    # 电压上下限 = 8 × 单体上下限
    v_max_v = 3.65 * series_count     # 29.2 V
    v_min_v = 2.50 * series_count     # 20.0 V

    # 等效元件内阻 (模块级, 给定): R0 = 10 mΩ
    r0_ohm = 10.0e-3
    # RC 极化时间常数 (给定): tau ≈ 380 s
    tau_s = 380.0
    # rc (极化电阻 R1) 未知 -> 经验值放缩等效 8S (见 rc_default 说明)
    #   经验放缩: 单体 R0_cell=0.3mΩ, 取 R1_cell≈R0_cell (中性假设);
    #            8S 串联 R1 = 8×R1_cell, 另加模块辅助极化 (与辅助欧姆电阻同比)
    #            => R1_8S ≈ R0_8S = 10 mΩ。故默认 rc = R0。
    rc_ohm = 10.0e-3

    # 支路间连接电阻 (母线/接触器/熔断器接触), 来自参考 PACK_CONFIG
    r_conn_ohm = 5.0e-4

    # 热模型参数 (简化集中参数, 与参考一致的量级)
    ambient_temp_k = 298.15
    module_mass_kg = 8 * 5.4          # 8 颗单体质量
    module_cp_j_kgk = 1040.0
    module_surface_area_m2 = 8 * 0.13  # 8 颗表面积
    htc_w_m2k = 7.0

    @property
    def c1_farad(self):
        """极化电容 C1 = tau / R1 (F)。"""
        r1 = self.rc_ohm if self.rc_ohm > 0 else 1e-9
        return self.tau_s / r1

    @property
    def r_branch_ohm(self):
        """每支路等效串联总电阻 (DC) = R0 + R1。"""
        return self.r0_ohm + self.rc_ohm

    def pack_current(self, c_rate: float) -> float:
        """整包负载电流 (A) = c_rate × 单芯容量 × 并联数。"""
        return c_rate * self.cell_capacity_ah * self.parallel_count

    def summary_lines(self):
        return [
            ("化学体系", "LFP / 石墨 (等效)"),
            ("等效拓扑", f"1S{self.parallel_count}P (8S 压缩为 1 个 8×电压元件)"),
            ("支路元件容量", f"{self.cell_capacity_ah:.0f} Ah"),
            ("支路标称电压", f"{self.nominal_voltage_v:.1f} V"),
            ("支路电压范围", f"{self.v_min_v:.1f} ~ {self.v_max_v:.1f} V"),
            ("R0 (模块内阻)", f"{self.r0_ohm*1e3:.1f} mΩ"),
            ("tau (RC 时间常数)", f"{self.tau_s:.0f} s"),
            ("rc (极化电阻 R1)", f"{self.rc_ohm*1e3:.1f} mΩ (经验放缩)"),
            ("C1 (极化电容)", f"{self.c1_farad:.2e} F"),
            ("支路间连接电阻", f"{self.r_conn_ohm*1e3:.2f} mΩ"),
            ("整包容量", f"{self.cell_capacity_ah*self.parallel_count:.0f} Ah"),
            ("整包标称电压", f"{self.nominal_voltage_v:.1f} V"),
            ("整包能量", f"{self.nominal_voltage_v*self.cell_capacity_ah*self.parallel_count/1000:.2f} kWh"),
        ]


# ================================================================
# 2. 仿真结果容器
# ================================================================
class SimResult:
    def __init__(self):
        self.method = "equivalent-ecm"
        self.model_type = "等效1S2P 一阶RC"
        self.time_s = None
        self.pack_voltage_v = None
        self.pack_current_a = None
        self.branch_voltages_v = None     # (n_p, n_time)
        self.branch_currents_a = None     # (n_p, n_time)
        self.branch_socs = None           # (n_p, n_time)
        self.branch_temps_k = None        # (n_p, n_time)
        self.circulating_current_a = None  # (n_time,)
        self.branch_deviation_pct = None   # (n_p, n_time)
        self.stagger_join_s = None
        # 指标
        self.pack_capacity_ah = 0.0
        self.pack_energy_wh = 0.0
        self.mean_temp_rise_k = 0.0
        self.max_temp_rise_k = 0.0
        self.voltage_spread_v = 0.0
        self.circulating_current_max_a = 0.0
        self.circulating_current_mean_a = 0.0
        self.branch_deviation_max_pct = 0.0
        self.branch_deviation_mean_pct = 0.0

    def summary(self) -> str:
        L = [
            f"仿真方法:     {self.method}",
            f"电路模型:     {self.model_type}",
            f"仿真时长:     {self.time_s[-1]/3600:.2f} h",
            f"数据点数:     {len(self.time_s)}",
            "",
            f"电池包容量:   {self.pack_capacity_ah:.1f} Ah",
            f"电池包能量:   {self.pack_energy_wh:.1f} Wh ({self.pack_energy_wh/1000:.2f} kWh)",
            f"平均温升:     {self.mean_temp_rise_k:.2f} K",
            f"最大温升:     {self.max_temp_rise_k:.2f} K",
            f"电压极差:     {self.voltage_spread_v*1000:.1f} mV",
            f"环流最大值:   {self.circulating_current_max_a:.2f} A",
            f"环流均值:     {self.circulating_current_mean_a:.2f} A",
            f"最大偏流率:   {self.branch_deviation_max_pct:.2f} %",
            f"平均偏流率:   {self.branch_deviation_mean_pct:.2f} %",
        ]
        if self.stagger_join_s is not None:
            L.append(f"支路2并入时刻: {self.stagger_join_s/3600:.3f} h")
        return "\n".join(L)


# ================================================================
# 3. 核心仿真: 等效 1S2P 一阶 RC, 交错并入
# ================================================================
def simulate(c_rate: float = 0.5, initial_soc: float = 1.0, stagger_h: float = 0.25,
             cfg: EquivConfig = None, dt: float = 10.0, thermal: bool = True,
             max_h: float = None):
    """运行等效 1S2P 仿真。

    返回 SimResult。
    """
    if cfg is None:
        cfg = EquivConfig()
    # 自适应最大时长: 整包在 c_rate 下理论最短放电时长 = 1/c_rate 小时,
    # 低倍率 (如 0.2C) 需要更久, 取 max(4h, 1.6/c_rate) 留足余量, 避免被截断。
    if max_h is None:
        max_h = max(4.0, 1.6 / c_rate)
    n_p = cfg.parallel_count
    Q = cfg.cell_capacity_ah
    I_pack = cfg.pack_current(c_rate)        # 阶段1 支路1独担 / 阶段2 整包
    I_nom = I_pack / n_p                      # 阶段2 每支路标称电流
    T1 = float(stagger_h) * 3600.0
    R0 = cfg.r0_ohm
    R1 = cfg.rc_ohm
    C1 = cfg.c1_farad
    Rb = cfg.r_branch_ohm
    Rc = cfg.r_conn_ohm
    Vmin = cfg.v_min_v
    tau = cfg.tau_s

    print(f"  [等效1S2P] c_rate={c_rate}C, I_pack={I_pack:.1f}A, I_nom={I_nom:.1f}A")
    print(f"  [等效1S2P] R0={R0*1e3:.1f}mΩ, R1={R1*1e3:.1f}mΩ, tau={tau:.0f}s, C1={C1:.2e}F")
    print(f"  [等效1S2P] 支路1 单独工作 {stagger_h:.3f}h ({T1:.0f}s) 后支路2 并入")

    # 状态
    soc = np.array([initial_soc, initial_soc], dtype=float)
    vrc = np.array([0.0, 0.0], dtype=float)
    connected = [True, False]          # 支路2 在 T1 前未接入
    temp = np.array([cfg.ambient_temp_k, cfg.ambient_temp_k], dtype=float)
    m = cfg.module_mass_kg
    cp = cfg.module_cp_j_kgk
    A = cfg.module_surface_area_m2
    htc = cfg.htc_w_m2k
    Tamb = cfg.ambient_temp_k

    # 时间网格
    n_steps = int(round(max_h * 3600.0 / dt)) + 1

    # 结果缓冲
    T = np.zeros(n_steps)
    I1 = np.zeros(n_steps)
    I2 = np.zeros(n_steps)
    Vp = np.zeros(n_steps)
    V1 = np.zeros(n_steps)
    V2 = np.zeros(n_steps)
    S1 = np.zeros(n_steps)
    S2 = np.zeros(n_steps)
    T1r = np.zeros(n_steps)
    T2r = np.zeros(n_steps)
    Ic = np.zeros(n_steps)
    D1 = np.zeros(n_steps)
    D2 = np.zeros(n_steps)

    cut = [False, False]   # 触底截止标记

    def branch_voltage(b):
        return ocv_equiv(soc[b]) - (I_b[b] * R0) - vrc[b]

    n_used = 0
    join_recorded = False
    for k in range(n_steps):
        t = k * dt
        T[k] = t

        # 支路2 在 T1 接入 (若未触底)
        if (not connected[1]) and (not cut[1]) and t >= T1:
            connected[1] = True

        I_b = [0.0, 0.0]
        # 当前哪些支路处于并联放电活跃态
        both = connected[0] and connected[1]

        if t < T1:
            # 阶段1: 仅支路1 独担整包负载; 支路2 开路静置
            if connected[0]:
                I_b[0] = I_pack
            I_b[1] = 0.0
            Ic_circ = 0.0
        else:
            if both:
                E1 = ocv_equiv(soc[0]) - vrc[0]
                E2 = ocv_equiv(soc[1]) - vrc[1]
                denom = 2.0 * Rb + 2.0 * Rc
                Ic_circ = (E1 - E2) / denom
                I_b[0] = I_nom + Ic_circ
                I_b[1] = I_nom - Ic_circ
            elif connected[0]:
                I_b[0] = I_pack
                I_b[1] = 0.0
                Ic_circ = 0.0
            elif connected[1]:
                I_b[1] = I_pack
                I_b[0] = 0.0
                Ic_circ = 0.0
            else:
                I_b = [0.0, 0.0]
                Ic_circ = 0.0

        # 记录
        I1[k], I2[k] = I_b[0], I_b[1]
        Ic[k] = Ic_circ
        S1[k], S2[k] = soc[0], soc[1]
        T1r[k], T2r[k] = temp[0], temp[1]

        # 整包电压/电流
        if connected[0]:
            vb0 = ocv_equiv(soc[0]) - I_b[0] * R0 - vrc[0]
            Vp[k] = vb0
            V1[k] = vb0
        else:
            vb1 = ocv_equiv(soc[1]) - I_b[1] * R0 - vrc[1]
            Vp[k] = vb1
            V1[k] = vb1
        # 支路2 端电压 (即使开路也记录其开路电压)
        if connected[1]:
            V2[k] = ocv_equiv(soc[1]) - I_b[1] * R0 - vrc[1]
        else:
            V2[k] = ocv_equiv(soc[1])   # 开路 OCV

        # 偏流率
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = Ic_circ / I_nom * 100.0 if abs(I_nom) > 1e-6 else 0.0
        D1[k], D2[k] = d1, -d1

        # 动力学更新
        for b in range(n_p):
            if connected[b] and abs(I_b[b]) > 1e-9:
                # RC 极化电压更新 (一阶)
                vrc[b] += dt * (I_b[b] * R1 - vrc[b]) / tau
                # SOC 更新
                soc[b] -= I_b[b] * dt / (Q * 3600.0)
                soc[b] = max(soc[b], 0.0)
                # 热: 仅电芯内部欧姆损产热 (8×0.3mΩ); 10mΩ 模块电阻多为
                # 母线/接触器损, 不在电芯热容内
                if thermal:
                    r_heat = cfg.series_count * 0.3e-3
                    P = I_b[b] ** 2 * r_heat
                    temp[b] += dt * (P - htc * A * (temp[b] - Tamb)) / (m * cp)
                # 触底截止 (放电且端电压≤下限 或 SOC≤0)
                vb = ocv_equiv(soc[b]) - I_b[b] * R0 - vrc[b]
                if I_b[b] > 0 and (vb <= Vmin or soc[b] <= 1e-4):
                    connected[b] = False
                    cut[b] = True
            else:
                # 开路支路: RC 衰减, SOC 冻结, 冷却
                vrc[b] += dt * (0.0 - vrc[b]) / tau
                if thermal:
                    temp[b] += dt * (0.0 - htc * A * (temp[b] - Tamb)) / (m * cp)

        # 两条支路都断开 -> 结束
        if (not connected[0]) and (not connected[1]):
            break

    # 裁剪 (自然结束 k=n_steps-1; break 时 k=断开步, 含该点)
    n = k + 1
    T = T[:n]; I1 = I1[:n]; I2 = I2[:n]; Vp = Vp[:n]; V1 = V1[:n]; V2 = V2[:n]
    S1 = S1[:n]; S2 = S2[:n]; T1r = T1r[:n]; T2r = T2r[:n]; Ic = Ic[:n]; D1 = D1[:n]; D2 = D2[:n]

    # ---- 指标 ----
    dt_arr = np.diff(T, prepend=0.0)
    pack_current_total = I1 + I2
    pack_capacity = np.sum(np.abs(pack_current_total) * dt_arr) / 3600.0
    pack_energy = np.sum(np.abs(Vp * pack_current_total) * dt_arr) / 3600.0

    mean_temp_rise = float(np.mean([temp[0] - cfg.ambient_temp_k, temp[1] - cfg.ambient_temp_k]))
    max_temp_rise = float(max(temp[0], temp[1]) - cfg.ambient_temp_k)

    # 电压极差: 取稳定并联段 (两支路均处于 SOC 平台区, 剔除并入瞬间/
    # 高SOC暂态), 反映稳态支路不平衡; 若取不到则退回全程并联活跃段
    both_mask = (np.abs(I1) > 1.0) & (np.abs(I2) > 1.0)
    if both_mask.any():
        steady = both_mask & (S1 < 0.85) & (S2 < 0.85)
        if steady.any():
            spread = float(np.max(np.abs(V1[steady] - V2[steady])))
        else:
            spread = float(np.max(np.abs(V1[both_mask] - V2[both_mask])))
    else:
        spread = 0.0

    # 环流/偏流仅在两支路均放电活跃段评估
    active = both_mask
    if active.any():
        cc_max = float(np.max(np.abs(Ic[active])))
        cc_mean = float(np.mean(np.abs(Ic[active])))
        bd_max = float(np.max(np.abs(D1[active])))
        bd_mean = float(np.mean(np.abs(D1[active])))
    else:
        cc_max = cc_mean = bd_max = bd_mean = 0.0

    res = SimResult()
    res.time_s = T
    res.pack_voltage_v = Vp
    res.pack_current_a = pack_current_total
    res.branch_voltages_v = np.vstack([V1, V2])
    res.branch_currents_a = np.vstack([I1, I2])
    res.branch_socs = np.vstack([S1, S2])
    res.branch_temps_k = np.vstack([T1r, T2r])
    res.circulating_current_a = Ic
    res.branch_deviation_pct = np.vstack([D1, D2])
    res.stagger_join_s = T1
    res.pack_capacity_ah = pack_capacity
    res.pack_energy_wh = pack_energy
    res.mean_temp_rise_k = mean_temp_rise
    res.max_temp_rise_k = max_temp_rise
    res.voltage_spread_v = spread
    res.circulating_current_max_a = cc_max
    res.circulating_current_mean_a = cc_mean
    res.branch_deviation_max_pct = bd_max
    res.branch_deviation_mean_pct = bd_mean

    print(f"  [等效1S2P] 仿真完成: 时长 {T[-1]/3600:.2f}h, 数据点 {n}")
    print(f"  [等效1S2P] 容量 {pack_capacity:.1f}Ah, 能量 {pack_energy/1000:.2f}kWh")
    print(f"  [等效1S2P] 并入后环流峰值 {cc_max:.2f}A, 最大偏流率 {bd_max:.2f}%")
    return res


# ================================================================
# 4. 结果保存
# ================================================================
def save_results(result: SimResult, output_dir: str):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "Time [s]": result.time_s,
        "Time [h]": result.time_s / 3600,
        "Pack Voltage [V]": result.pack_voltage_v,
        "Pack Current [A]": result.pack_current_a,
        "Branch1_Voltage [V]": result.branch_voltages_v[0],
        "Branch2_Voltage [V]": result.branch_voltages_v[1],
        "Branch1_Current [A]": result.branch_currents_a[0],
        "Branch2_Current [A]": result.branch_currents_a[1],
        "Branch1_SOC": result.branch_socs[0],
        "Branch2_SOC": result.branch_socs[1],
        "Branch1_Temp [K]": result.branch_temps_k[0],
        "Branch2_Temp [K]": result.branch_temps_k[1],
        "Circulating_Current [A]": result.circulating_current_a,
        "Branch1_Deviation [%]": result.branch_deviation_pct[0],
        "Branch2_Deviation [%]": result.branch_deviation_pct[1],
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "simulation_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  CSV 数据: {csv_path}")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("314Ah LFP 8S2P 等效 1S2P 降阶模型仿真摘要\n")
        f.write("Equivalent 1S2P (8S compressed) Reduced-Order Simulation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(result.summary())
        f.write("\n\n")
        f.write("-" * 60 + "\n")
        f.write("等效元件参数 (Equivalent 8S Element)\n")
        f.write("-" * 60 + "\n")
        cfg_lines = EquivConfig().summary_lines()
        for k, v in cfg_lines:
            f.write(f"  {k}: {v}\n")
    print(f"  摘要文件: {summary_path}")


# ================================================================
# 5. 绘图 (6 张, 与参考项目一致)
# ================================================================
def plot_pack_results(result, save_path=None):
    t = result.time_s / 3600
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(t, result.pack_voltage_v, 'b-', linewidth=2, label='整包电压 [V]')
    ax1.set_xlabel("时间 [h]"); ax1.set_ylabel("整包电压 [V]", color='b')
    ax1.tick_params(axis='y', labelcolor='b'); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, result.pack_current_a, 'r-', linewidth=2, label='整包电流 [A]')
    ax2.set_ylabel("整包电流 [A]", color='r'); ax2.tick_params(axis='y', labelcolor='r')
    if result.stagger_join_s is not None:
        jh = result.stagger_join_s / 3600
        ax1.axvline(x=jh, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.text(jh, ax1.get_ylim()[1], ' 支路2并入', color='purple', fontsize=10, fontweight='bold', va='top')
    plt.title("等效 1S2P 整包电压 / 电流 (Pack Voltage & Current)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  整包结果图: {save_path}")
    return fig


def plot_circulating_current(result, save_path=None):
    n_p = 2
    t = result.time_s / 3600
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("等效 1S2P 环流 / 偏流评估 (Circulating & Branch Imbalance)", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t, result.branch_currents_a[0], 'b-', linewidth=2, label='支路1 电流')
    ax.plot(t, result.branch_currents_a[1], 'g-', linewidth=2, label='支路2 电流')
    ax.plot(t, result.pack_current_a / n_p, 'k--', linewidth=1.2, alpha=0.7, label='标称支路电流 (2并联均分)')
    ax.set_xlabel("时间 [h]"); ax.set_ylabel("电流 [A]")
    ax.set_title("并联支路电流", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

    if result.stagger_join_s is not None:
        jh = result.stagger_join_s / 3600
        for _ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            _ax.axvline(x=jh, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0, 0].text(jh, axes[0, 0].get_ylim()[1], ' 支路2并入', color='purple', fontsize=9, fontweight='bold', va='top')

    ax = axes[0, 1]
    ax.plot(t, result.circulating_current_a, 'r-', linewidth=2)
    ax.fill_between(t, result.circulating_current_a, alpha=0.1, color='red')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.6)
    ax.set_xlabel("时间 [h]"); ax.set_ylabel("环流 [A]")
    ax.set_title(f"并联回路环流 (峰值 {result.circulating_current_max_a:.2f} A)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, result.branch_deviation_pct[0], 'b-', linewidth=2, label='支路1 偏流率')
    ax.plot(t, result.branch_deviation_pct[1], 'g-', linewidth=2, label='支路2 偏流率')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.6)
    ax.set_xlabel("时间 [h]"); ax.set_ylabel("偏流率 [%]")
    ax.set_title(f"支路偏流率 (峰值 {result.branch_deviation_max_pct:.2f} %)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]; ax.axis('off')
    txt = (
        f"┌─────────────────────────────────┐\n"
        f"│   环流 / 偏流评估 (Summary)      │\n"
        f"├─────────────────────────────────┤\n"
        f"│ 等效拓扑: 1S{n_p}P (8S→1元件)     │\n"
        f"│ 仿真方法: {result.method:<16s} │\n"
        f"├─────────────────────────────────┤\n"
        f"│ 环流最大值: {result.circulating_current_max_a:>8.2f} A   │\n"
        f"│ 环流均值:   {result.circulating_current_mean_a:>8.2f} A   │\n"
        f"│ 最大偏流率: {result.branch_deviation_max_pct:>8.2f} %   │\n"
        f"│ 平均偏流率: {result.branch_deviation_mean_pct:>8.2f} %   │\n"
        f"├─────────────────────────────────┤\n"
        f"│ 环流 = (E1-E2)/(R1+R2+2Rc)      │\n"
        f"│ E: 支路OCV  R: 支路等效电阻      │\n"
    )
    if result.stagger_join_s is not None:
        txt += f"│ 支路2并入: {result.stagger_join_s/3600:.3f} h         │\n"
    txt += f"└─────────────────────────────────┘"
    ax.text(0.05, 0.5, txt, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#fff5ee', alpha=0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  环流偏流图: {save_path}")
    return fig


def plot_voltage_distribution(result, save_path=None):
    t = result.time_s / 3600
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, result.branch_voltages_v[0], 'b-', linewidth=2, label='支路1 端电压 [V]')
    ax.plot(t, result.branch_voltages_v[1], 'g-', linewidth=2, label='支路2 端电压 [V]')
    ax.plot(t, result.pack_voltage_v, 'k--', linewidth=1.5, alpha=0.8, label='整包电压 [V]')
    ax.axhline(y=EquivConfig().v_min_v, color='r', linestyle=':', alpha=0.6, label=f'放电截止 {EquivConfig().v_min_v:.0f}V')
    if result.stagger_join_s is not None:
        jh = result.stagger_join_s / 3600
        ax.axvline(x=jh, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel("时间 [h]"); ax.set_ylabel("电压 [V]")
    ax.set_title("支路端电压分布 (Branch Terminal Voltages)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  电压分布图: {save_path}")
    return fig


def plot_branch_consistency(result, save_path=None):
    t = result.time_s / 3600
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, result.branch_socs[0] * 100, 'b-', linewidth=2, label='支路1 SOC [%]')
    ax.plot(t, result.branch_socs[1] * 100, 'g-', linewidth=2, label='支路2 SOC [%]')
    ax.fill_between(t, (result.branch_socs[0] - result.branch_socs[1]) * 100,
                    alpha=0.15, color='orange', label='SOC 差 (支路1-支路2)')
    if result.stagger_join_s is not None:
        jh = result.stagger_join_s / 3600
        ax.axvline(x=jh, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(jh, 100, ' 支路2并入', color='purple', fontsize=9, fontweight='bold', va='top')
    ax.set_xlabel("时间 [h]"); ax.set_ylabel("SOC [%]")
    ax.set_title("支路一致性 / SOC 发散 (Branch SOC Divergence)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  一致性图: {save_path}")
    return fig


def plot_pack_topology(save_path=None):
    cfg = EquivConfig()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    # 画等效 1S2P 拓扑
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.6, "等效 1S2P 拓扑 (8S 压缩为 1 个 8×电压元件)", fontsize=14, fontweight='bold', ha='center')
    # 母线
    ax.plot([1, 9], [5, 5], 'k-', linewidth=3)
    ax.plot([1, 9], [1, 1], 'k-', linewidth=3)
    ax.text(0.5, 5, "Busbar+", fontsize=11, ha='right', va='center')
    ax.text(0.5, 1, "Busbar-", fontsize=11, ha='right', va='center')
    # 支路1
    ax.add_patch(plt.Rectangle((2.5, 2.2), 2.0, 1.6, fill=True, facecolor='#cfe8ff', edgecolor='b', linewidth=2))
    ax.text(3.5, 3.6, "支路1 (等效8S)", fontsize=10, ha='center', fontweight='bold')
    ax.text(3.5, 3.0, "314Ah / 25.6V", fontsize=9, ha='center')
    ax.text(3.5, 2.6, f"R0={cfg.r0_ohm*1e3:.0f}mΩ τ={cfg.tau_s:.0f}s", fontsize=8, ha='center')
    ax.plot([3.5, 3.5], [5, 3.8], 'b-', linewidth=2)
    ax.plot([3.5, 3.5], [2.2, 1], 'b-', linewidth=2)
    # 支路2
    ax.add_patch(plt.Rectangle((5.5, 2.2), 2.0, 1.6, fill=True, facecolor='#d6f5d6', edgecolor='g', linewidth=2))
    ax.text(6.5, 3.6, "支路2 (等效8S)", fontsize=10, ha='center', fontweight='bold')
    ax.text(6.5, 3.0, "314Ah / 25.6V", fontsize=9, ha='center')
    ax.text(6.5, 2.6, f"R0={cfg.r0_ohm*1e3:.0f}mΩ τ={cfg.tau_s:.0f}s", fontsize=8, ha='center')
    ax.plot([6.5, 6.5], [5, 3.8], 'g-', linewidth=2)
    ax.plot([6.5, 6.5], [2.2, 1], 'g-', linewidth=2)
    ax.text(5, 0.4, "2 条并联支路, 每支路 = 8 颗串联单体等效成 1 个元件", fontsize=10, ha='center', style='italic')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  拓扑图: {save_path}")
    return fig


def plot_ocv_curve(save_path=None):
    soc_q = np.linspace(0, 1, 200)
    v_cell = ocv_cell(soc_q)
    v_eq = ocv_equiv(soc_q)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(soc_q * 100, v_cell, 'b-', linewidth=2)
    ax1.set_xlabel("SOC [%]"); ax1.set_ylabel("单体 OCV [V]")
    ax1.set_title("单体 (314Ah LFP) OCV-SOC", fontsize=12, fontweight='bold'); ax1.grid(True, alpha=0.3)
    ax2.plot(soc_q * 100, v_eq, 'r-', linewidth=2)
    ax2.axhline(y=EquivConfig().v_min_v, color='k', linestyle=':', alpha=0.6)
    ax2.axhline(y=EquivConfig().v_max_v, color='k', linestyle=':', alpha=0.6)
    ax2.set_xlabel("SOC [%]"); ax2.set_ylabel("等效8S OCV [V]")
    ax2.set_title("等效 8S 元件 OCV-SOC (×8)", fontsize=12, fontweight='bold'); ax2.grid(True, alpha=0.3)
    fig.suptitle("OCV-SOC 曲线 (源自参考项目 PyBaMM/Prada2013 LFP)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  OCV曲线图: {save_path}")
    return fig


# ================================================================
# 6. 主程序 / CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="314Ah LFP 8S2P 等效 1S2P 降阶模型仿真")
    parser.add_argument("--crate", type=float, default=0.5, help="C-rate (默认 0.5C)")
    parser.add_argument("--initial-soc", type=float, default=1.0, help="初始 SOC (默认 1.0)")
    parser.add_argument("--stagger", type=float, default=0.25,
                        help="交错并入: 支路1单独工作 N 小时后支路2并入 (小时, 默认 0.25)")
    parser.add_argument("--r0", type=float, default=10.0, help="等效8S元件 R0 [mΩ] (默认 10)")
    parser.add_argument("--rc", type=float, default=10.0, help="等效8S元件 极化电阻 R1 [mΩ] (默认 10, 经验放缩)")
    parser.add_argument("--tau", type=float, default=380.0, help="RC 时间常数 tau [s] (默认 380)")
    parser.add_argument("--dt", type=float, default=10.0, help="积分步长 [s] (默认 10)")
    parser.add_argument("--no-thermal", action="store_true", help="禁用热模型")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    parser.add_argument("--no-show", action="store_true", help="不弹窗 (仅保存)")
    args = parser.parse_args()

    cfg = EquivConfig()
    cfg.r0_ohm = args.r0 * 1e-3
    cfg.rc_ohm = args.rc * 1e-3
    cfg.tau_s = args.tau

    print("\n" + "=" * 65)
    print("  314Ah LFP 8S2P 等效 1S2P 降阶模型仿真")
    print("=" * 65)
    for k, v in cfg.summary_lines():
        print(f"  {k}: {v}")
    print(f"\n  C-rate:        {args.crate}C")
    print(f"  初始 SOC:      {args.initial_soc}")
    print(f"  交错并入:      支路1工作 {args.stagger:.3f}h 后支路2并入")
    print(f"  积分步长:      {args.dt}s")
    print(f"  热模型:        {'启用' if not args.no_thermal else '禁用'}")

    result = simulate(c_rate=args.crate, initial_soc=args.initial_soc, stagger_h=args.stagger,
                      cfg=cfg, dt=args.dt, thermal=not args.no_thermal)

    # 保存
    save_results(result, args.output)

    # 绘图
    od = args.output
    plot_pack_results(result, os.path.join(od, "pack_results.png"))
    plot_circulating_current(result, os.path.join(od, "circulating_current.png"))
    plot_voltage_distribution(result, os.path.join(od, "voltage_distribution.png"))
    plot_branch_consistency(result, os.path.join(od, "branch_consistency.png"))
    plot_pack_topology(os.path.join(od, "pack_topology.png"))
    plot_ocv_curve(os.path.join(od, "ocv_curve.png"))

    # 打印摘要
    print("\n" + "=" * 60)
    print("  仿真摘要")
    print("=" * 60)
    print(result.summary())

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
