# -*- coding: utf-8 -*-
"""
等效 1S2P 交错并入 (staggered join) 参数扫描
================================================
固定等效 8S 元件电气参数 (R0/rc/tau), 扫描「支路1 单独工作多长时间后支路2并入」
(stagger_h) 从 0.05h 到 2h, 每个工况独立仿真并把 6 张图 + CSV + 摘要写入
扫描结果/stagger_X.XXh/ 子目录; 顶层汇总 scan_summary.csv + scan_comparison.png。

用法:
    python scan_stagger.py            # 默认扫描 0.05~2h
    python scan_stagger.py --r0 3 --rc 3 --tau 105
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用主工程的仿真与绘图能力
import equivalent_1s2p_sim as E

# 扫描网格: 0.05h ~ 2h
DEFAULT_STAGGER = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]


def fmt_h(h: float) -> str:
    """子目录名: stagger_0.05h / stagger_0.25h / stagger_2.00h ..."""
    return f"stagger_{h:.2f}h"


def run_one(stagger_h: float, r0_mohm: float, rc_mohm: float, tau_s: float,
            out_dir: str, crate=0.5, initial_soc=1.0, dt=10.0):
    cfg = E.EquivConfig()
    cfg.r0_ohm = r0_mohm * 1e-3
    cfg.rc_ohm = rc_mohm * 1e-3
    cfg.tau_s = tau_s

    res = E.simulate(c_rate=crate, initial_soc=initial_soc, stagger_h=stagger_h,
                     cfg=cfg, dt=dt, thermal=True)

    # 保存该工况的 6 图 + CSV + 摘要
    E.save_results(res, out_dir)
    E.plot_pack_results(res, os.path.join(out_dir, "pack_results.png"))
    E.plot_circulating_current(res, os.path.join(out_dir, "circulating_current.png"))
    E.plot_voltage_distribution(res, os.path.join(out_dir, "voltage_distribution.png"))
    E.plot_branch_consistency(res, os.path.join(out_dir, "branch_consistency.png"))
    E.plot_pack_topology(os.path.join(out_dir, "pack_topology.png"))
    E.plot_ocv_curve(os.path.join(out_dir, "ocv_curve.png"))
    plt.close("all")

    # 是否成功并入 (支路2 在 T1 后真正带过电流)
    T1 = stagger_h * 3600.0
    joined_mask = (np.abs(res.branch_currents_a[1]) > 1.0) & (res.time_s >= T1)
    joined = bool(np.any(joined_mask))

    T = res.time_s / 3600.0
    s1_end = res.branch_socs[0][-1] if len(res.branch_socs[0]) else float("nan")
    s2_end = res.branch_socs[1][-1] if len(res.branch_socs[1]) else float("nan")

    return {
        "stagger_h": stagger_h,
        "duration_h": float(T[-1]),
        "joined": joined,
        "pack_capacity_ah": res.pack_capacity_ah,
        "pack_energy_wh": res.pack_energy_wh,
        "circ_peak_a": res.circulating_current_max_a,
        "circ_mean_a": res.circulating_current_mean_a,
        "dev_peak_pct": res.branch_deviation_max_pct,
        "dev_mean_pct": res.branch_deviation_mean_pct,
        "voltage_spread_mv": res.voltage_spread_v * 1000.0,
        "mean_temp_k": res.mean_temp_rise_k,
        "max_temp_k": res.max_temp_rise_k,
        "branch1_soc_end": s1_end * 100.0,
        "branch2_soc_end": s2_end * 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=float, default=3.0, help="等效8S R0 [mΩ]")
    ap.add_argument("--rc", type=float, default=3.0, help="等效8S 极化电阻 R1 [mΩ]")
    ap.add_argument("--tau", type=float, default=105.0, help="RC 时间常数 [s]")
    ap.add_argument("--crate", type=float, default=0.5)
    ap.add_argument("--stagger-list", type=str, default="",
                    help="逗号分隔的 stagger 值 (小时); 留空用默认网格")
    ap.add_argument("--root", type=str, default="扫描结果", help="扫描结果根目录")
    args = ap.parse_args()

    if args.stagger_list:
        stagger_list = [float(x) for x in args.stagger_list.split(",") if x.strip()]
    else:
        stagger_list = DEFAULT_STAGGER

    root = args.root
    os.makedirs(root, exist_ok=True)

    print("\n" + "=" * 65)
    print("  等效 1S2P 交错并入参数扫描")
    print("=" * 65)
    print(f"  固定参数: R0={args.r0}mΩ, rc={args.rc}mΩ, tau={args.tau}s, "
          f"{args.crate}C, 初始SOC=1.0")
    print(f"  扫描 stagger: {stagger_list}")
    print(f"  输出根目录: {root}\n")

    rows = []
    for sh in stagger_list:
        sub = os.path.join(root, fmt_h(sh))
        print(f">>> stagger = {sh:.2f}h  ->  {sub}")
        row = run_one(sh, args.r0, args.rc, args.tau, sub,
                      crate=args.crate)
        rows.append(row)
        print(f"    容量 {row['pack_capacity_ah']:.1f}Ah | 环流峰值 "
              f"{row['circ_peak_a']:.1f}A | 并入={'是' if row['joined'] else '否(提前截止)'} "
              f"| 时长 {row['duration_h']:.2f}h\n")

    # ---- 汇总 CSV ----
    import pandas as pd
    cols = ["stagger_h", "duration_h", "joined", "pack_capacity_ah",
            "pack_energy_wh", "circ_peak_a", "circ_mean_a", "dev_peak_pct",
            "dev_mean_pct", "voltage_spread_mv", "mean_temp_k", "max_temp_k",
            "branch1_soc_end", "branch2_soc_end"]
    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(root, "scan_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"扫描汇总 CSV: {csv_path}")

    # ---- 控制台表格 ----
    print("\n" + "=" * 95)
    print("  扫描结果汇总 (Stagger Scan Summary)")
    print("=" * 95)
    hdr = (f"{'stagger[h]':>10} {'并入':>5} {'时长[h]':>8} {'容量[Ah]':>9} "
           f"{'能量[kWh]':>9} {'环流峰[A]':>9} {'偏流峰[%]':>9} {'B1终SOC[%]':>10} {'B2终SOC[%]':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['stagger_h']:>10.2f} {'是' if r['joined'] else '否':>5} "
              f"{r['duration_h']:>8.2f} {r['pack_capacity_ah']:>9.1f} "
              f"{r['pack_energy_wh']/1000:>9.2f} {r['circ_peak_a']:>9.1f} "
              f"{r['dev_peak_pct']:>9.1f} {r['branch1_soc_end']:>10.1f} "
              f"{r['branch2_soc_end']:>10.1f}")
    print("=" * 95)

    # ---- 对比图 ----
    xs = np.array([r["stagger_h"] for r in rows])
    joined_arr = np.array([r["joined"] for r in rows])
    cap = np.array([r["pack_capacity_ah"] for r in rows])
    eng = np.array([r["pack_energy_wh"] for r in rows]) / 1000.0
    cpeak = np.array([r["circ_peak_a"] for r in rows])
    dpeak = np.array([r["dev_peak_pct"] for r in rows])
    dur = np.array([r["duration_h"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"等效 1S2P 交错并入扫描  (R0={args.r0}mΩ, rc={args.rc}mΩ, "
                 f"tau={args.tau}s, {args.crate}C)", fontsize=14, fontweight="bold")

    # 容量 / 能量
    ax = axes[0, 0]
    ax.plot(xs, cap, "o-", color="tab:blue", label="容量 [Ah]")
    ax.set_xlabel("支路1先行工作时间 [h]"); ax.set_ylabel("容量 [Ah]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(xs, eng, "s--", color="tab:orange", label="能量 [kWh]")
    ax2.set_ylabel("能量 [kWh]", color="tab:orange"); ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("电池包容量 / 能量 vs 先行时间")
    ax.grid(True, alpha=0.3)

    # 环流峰值
    ax = axes[0, 1]
    ax.plot(xs, cpeak, "o-", color="tab:red")
    ax.set_xlabel("支路1先行工作时间 [h]"); ax.set_ylabel("环流峰值 [A]")
    ax.set_title("环流峰值 vs 先行时间")
    for i, r in enumerate(rows):
        if not joined_arr[i]:
            ax.annotate("未并入", (xs[i], cpeak[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="gray")
    ax.grid(True, alpha=0.3)

    # 偏流率峰值
    ax = axes[1, 0]
    ax.plot(xs, dpeak, "o-", color="tab:green")
    ax.set_xlabel("支路1先行工作时间 [h]"); ax.set_ylabel("最大偏流率 [%]")
    ax.set_title("最大偏流率 vs 先行时间")
    ax.grid(True, alpha=0.3)

    # 总时长
    ax = axes[1, 1]
    ax.plot(xs, dur, "o-", color="tab:purple")
    ax.set_xlabel("支路1先行工作时间 [h]"); ax.set_ylabel("放电总时长 [h]")
    ax.set_title("放电总时长 vs 先行时间")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cmp_path = os.path.join(root, "scan_comparison.png")
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    print(f"扫描对比图: {cmp_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
