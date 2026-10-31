"""
EV Powertrain CAN Test Analyzer with Environmental Testing and Root Cause Analysis
----------------------------------------------------------------------------------
Covers: HV battery, DC-DC converter, onboard charger, HV motor + controller, HV fuse,
CAN protocol, environmental effects (cold / room / hot), test limits, root cause
analysis, Python / Pandas / scikit-learn and an Excel report.

IMPORTANT: all data is SIMULATED (no real vehicle). The test limits are illustrative
engineering limits, NOT clauses copied from a homologation standard. Say this in your
README and in interviews.

Run:  python ev_powertrain_test_analyzer.py
"""
import struct

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------ constants
N = 2700            # seconds per test run (1 Hz)
DRIVE_END = 2100    # drive 0-2100 s, plugged-in charging 2100-2700 s
CELLS, CAP_AH = 96, 60.0
FUSE_A = 250.0      # HV fuse rating (A)
CHG_A = 18.0        # charger rated current (A, pack side)

# test requirements (illustrative limits)
REQ = {
    "R1": "HV pack voltage within 300-420 V",
    "R2": "HV battery temperature <= 55 C",
    "R3": "HV pack current <= 250 A and HV fuse stays closed",
    "R4": "DC-DC LV output within 12.0-14.8 V",
    "R5": "Charger current <= temperature derating limit (+5%)",
    "R6": "Motor controller temperature <= 100 C",
    "R7": "No CAN message gap > 3 s on any node",
}
CAN_IDS = {0x100: "BMS", 0x200: "DC-DC", 0x300: "Charger", 0x400: "Motor controller"}


def derate(temp_c):
    """Charge-current derating vs battery temperature (cold protection)."""
    return np.where(temp_c < 0, 0.3, np.where(temp_c < 10, 0.6, 1.0))


# ------------------------------------------------------------------ 1. simulate
def simulate(ambient, faults=(), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)

    speed = np.clip(40 + 25 * np.sin(t / 120) + 10 * np.sin(t / 37) + rng.normal(0, 0.5, N), 0, 80)
    speed[DRIVE_END:] = 0
    v = speed / 3.6
    acc = np.gradient(v)
    p_mech = (1500 * acc * v + 0.5 * 1.2 * 0.7 * v ** 3 + 0.012 * 1500 * 9.81 * v) / 1000
    ptrac = np.clip(np.where(p_mech > 0, p_mech / 0.9, p_mech * 0.6), -30, 60)  # kW at battery
    rpm = v / 0.3 * 9 * 60 / (2 * np.pi)
    lv_i = 30 + 12 * np.sin(t / 200) + (20 if ambient < 5 else 0) + rng.normal(0, 1.5, N)  # A
    p_lv_hv = 13.8 * lv_i / 0.93 / 1000  # kW seen on the HV side

    soc = np.zeros(N); soc[0] = 0.9
    Tb = np.full(N, float(ambient)); Tc = np.full(N, ambient + 5.0)
    I = np.zeros(N); V = np.zeros(N); chg_i = np.zeros(N)
    fuse = np.ones(N)
    truth = np.array(["Normal"] * N, dtype=object)
    blown = False

    for i in range(N):
        prev = max(i - 1, 0)
        Rp = max(0.07 * (1 + 0.025 * (25 - Tb[prev])), 0.05)   # resistance rises when cold
        Vprev = V[prev] if i > 0 else CELLS * (3.0 + 1.2 * soc[0])

        if i >= DRIVE_END:
            taper = 1.0 if i < N - 250 else 0.2 + 0.8 * (N - i) / 250
            chg_i[i] = CHG_A * float(derate(Tb[prev])) * taper
            if "charger" in faults and 2300 <= i < 2360:
                chg_i[i] *= 1.5
                truth[i] = "Charger overcurrent"

        I[i] = (ptrac[i] + p_lv_hv[i]) * 1000 / Vprev - chg_i[i]

        if "short" in faults and 1700 <= i < 1703:           # short circuit event
            I[i] = 420 + rng.normal(0, 5)
            truth[i] = "HV short / fuse blown"
        if "short" in faults and i == 1703:
            blown = True
        if blown:
            fuse[i] = 0
            I[i] = 0.0
            chg_i[i] = 0.0
            ptrac[i] = 0.0
            rpm[i] = 0.0
            if truth[i] == "Normal":
                truth[i] = "HV short / fuse blown"

        soc[i] = soc[prev] - I[i] / (CAP_AH * 3600) if i > 0 else soc[0]
        V[i] = CELLS * (3.0 + 1.2 * soc[i]) - I[i] * Rp + rng.normal(0, 0.4)
        Tb[i] = Tb[prev] + (I[i] ** 2 * Rp) / 150000 - (Tb[prev] - ambient) * 0.002 + rng.normal(0, 0.01) if i > 0 else ambient
        if "cooling" in faults and 900 <= i < 1000:
            Tb[i] += 0.35                                    # cooling loop failure
            truth[i] = "Cooling failure"
        Tc[i] = Tc[prev] + (ambient + 8 + 1.2 * abs(ptrac[i]) - Tc[prev]) / 120 if i > 0 else Tc[0]

    omega = np.maximum(rpm * 2 * np.pi / 60, 1e-3)
    torque = np.clip(np.where(rpm > 50, ptrac * 1000 / omega, 0), -320, 320)

    dcdc_v = 13.8 - 0.002 * lv_i + rng.normal(0, 0.05, N)
    dcdc_t = ambient + 12 + 0.15 * lv_i
    if "dcdc" in faults:
        dcdc_v[500:540] = 11.2 + rng.normal(0, 0.05, 40)
        truth[500:540] = "DC-DC undervoltage"
    if "short" in faults:                                   # HV lost -> DC-DC cannot run
        dcdc_v[1703:DRIVE_END] = 11.8 + rng.normal(0, 0.05, DRIVE_END - 1703)

    chg_v = V + rng.normal(0, 0.2, N)
    status = (np.arange(N) >= DRIVE_END).astype(int)
    status[fuse == 0] = 0

    # ---- encode raw CAN frames (little-endian, scaled integers)
    frames = []
    for i in range(N):
        ts = float(i)
        frames.append((ts, 0x100, struct.pack("<HhhBB", int(V[i] * 10), int(I[i] * 10), int(Tb[i] * 10),
                                             int(soc[i] * 200), int(fuse[i]))))
        frames.append((ts, 0x200, struct.pack("<HHh", int(dcdc_v[i] * 100), int(lv_i[i] * 10), int(dcdc_t[i] * 10))))
        if i >= DRIVE_END:
            frames.append((ts, 0x300, struct.pack("<HHB", int(chg_v[i] * 10), int(chg_i[i] * 10), int(status[i]))))
        frames.append((ts, 0x400, struct.pack("<hhh", int(rpm[i]), int(torque[i] * 10), int(Tc[i] * 10))))
    if "can" in faults:                                     # motor controller node drops off the bus
        frames = [f for f in frames if not (f[1] == 0x400 and 1200 <= f[0] < 1210)]
        truth[1200:1210] = "CAN timeout (motor controller)"
    return frames, truth


# ------------------------------------------------------------------ 2. decode
def decode(frames):
    rows = {0x100: [], 0x200: [], 0x300: [], 0x400: []}
    for ts, cid, data in frames:
        if cid == 0x100:
            v, c, tp, s, f = struct.unpack("<HhhBB", data)
            rows[cid].append((ts, v / 10, c / 10, tp / 10, s / 2, f))
        elif cid == 0x200:
            v, c, tp = struct.unpack("<HHh", data)
            rows[cid].append((ts, v / 100, c / 10, tp / 10))
        elif cid == 0x300:
            v, c, s = struct.unpack("<HHB", data)
            rows[cid].append((ts, v / 10, c / 10, s))
        elif cid == 0x400:
            r, tq, tp = struct.unpack("<hhh", data)
            rows[cid].append((ts, r, tq / 10, tp / 10))
    bms = pd.DataFrame(rows[0x100], columns=["time_s", "pack_v", "pack_a", "batt_temp_c", "soc_pct", "fuse_closed"])
    dcdc = pd.DataFrame(rows[0x200], columns=["time_s", "dcdc_v", "dcdc_a", "dcdc_temp_c"])
    chg = pd.DataFrame(rows[0x300], columns=["time_s", "chg_v", "chg_a", "chg_status"])
    mc = pd.DataFrame(rows[0x400], columns=["time_s", "motor_rpm", "motor_torque_nm", "ctrl_temp_c"])
    df = bms.merge(dcdc, on="time_s", how="outer").merge(chg, on="time_s", how="outer").merge(mc, on="time_s", how="outer")
    return df.sort_values("time_s").reset_index(drop=True)


# ------------------------------------------------------------------ 3. test + root cause
def can_gaps(frames):
    out = {}
    for cid, name in CAN_IDS.items():
        ts = np.array(sorted(f[0] for f in frames if f[1] == cid))
        if len(ts) < 2:
            out[cid] = (0.0, np.nan)
            continue
        g = np.diff(ts)
        k = int(np.argmax(g))
        out[cid] = (float(g.max()), float(ts[k]))
    return out


def run_tests(run, df, frames):
    gaps = can_gaps(frames)
    fuse_ok = df["fuse_closed"].fillna(1) == 1
    masks = {
        "R1": ((df["pack_v"] < 300) | (df["pack_v"] > 420)) & fuse_ok,
        "R2": df["batt_temp_c"] > 55,
        "R3": (df["pack_a"].abs() > FUSE_A) | (df["fuse_closed"] == 0),
        "R4": (df["dcdc_v"] < 12.0) | (df["dcdc_v"] > 14.8),
        "R5": df["chg_a"] > CHG_A * derate(df["batt_temp_c"].fillna(25).values) * 1.05,
        "R6": df["ctrl_temp_c"] > 100,
    }
    res = []
    for rid, m in masks.items():
        m = m.fillna(False)
        first = df.loc[m, "time_s"].min() if m.any() else np.nan
        res.append({"run": run, "req": rid, "requirement": REQ[rid],
                    "result": "FAIL" if m.any() else "PASS",
                    "violating_samples": int(m.sum()), "first_violation_s": first})
    worst = max(g for g, _ in gaps.values())
    bad = [(cid, g, at) for cid, (g, at) in gaps.items() if g > 3.0]
    res.append({"run": run, "req": "R7", "requirement": REQ["R7"],
                "result": "FAIL" if bad else "PASS", "violating_samples": len(bad),
                "first_violation_s": min([at for _, _, at in bad]) if bad else np.nan})
    return res, masks, gaps


def root_causes(run, df, masks, gaps):
    out = []

    def add(req, t0, evidence, cause, action):
        out.append({"run": run, "req": req, "time_s": t0, "evidence": evidence,
                    "likely_root_cause": cause, "next_check": action})

    if masks["R2"].any():
        t0 = df.loc[masks["R2"], "time_s"].min()
        seg = df[(df["time_s"] >= t0 - 60) & (df["time_s"] <= t0)]
        mean_a = seg["pack_a"].abs().mean()
        slope = (seg["batt_temp_c"].iloc[-1] - seg["batt_temp_c"].iloc[0]) / 60
        if mean_a < 100 and slope > 0.05:
            add("R2", t0, f"Temp rose {slope:.2f} C/s in the 60 s before the limit while mean current was only {mean_a:.0f} A",
                "Heating not explained by load: cooling loop / pump failure or thermal sensor fault",
                "Check coolant flow, pump and fan commands, thermistor readings")
        else:
            add("R2", t0, f"Temp {slope:.2f} C/s with mean current {mean_a:.0f} A",
                "Load-induced (I2R) overheating", "Check current limits and cell resistance")

    if masks["R3"].any():
        t0 = df.loc[masks["R3"], "time_s"].min()
        peak = df["pack_a"].abs().max()
        add("R3", t0, f"Peak current {peak:.0f} A (> {FUSE_A:.0f} A rating), fuse state went open",
            "HV short circuit / overcurrent event blew the HV fuse",
            "Inspect HV harness, motor-controller DC link and insulation resistance before replacing the fuse")

    if masks["R4"].any():
        t0 = df.loc[masks["R4"], "time_s"].min()
        seg = df[masks["R4"]]
        fuse_open = (seg["fuse_closed"] == 0).mean() > 0.5
        pack_ok = seg["pack_v"].between(300, 420).mean() > 0.9
        low = seg["dcdc_v"].min()
        if fuse_open:
            add("R4", t0, f"DC-DC output fell to {low:.1f} V while the HV fuse was open",
                "Secondary effect: HV supply lost, so the DC-DC cannot run", "Fix the HV fuse/short fault first")
        elif pack_ok:
            add("R4", t0, f"DC-DC output fell to {low:.1f} V while HV pack voltage stayed normal",
                "Fault inside the DC-DC converter (output stage) or LV harness/fuse",
                "Measure DC-DC output under load and inspect the LV fuse and wiring")
        else:
            add("R4", t0, f"DC-DC low ({low:.1f} V) together with low HV pack voltage",
                "HV input supply problem", "Check HV battery voltage and contactors")

    if masks["R5"].any():
        t0 = df.loc[masks["R5"], "time_s"].min()
        seg = df[masks["R5"]]
        add("R5", t0, f"Charger current reached {seg['chg_a'].max():.1f} A at battery temp {seg['batt_temp_c'].mean():.0f} C, above the allowed limit",
            "Charger current control not honouring the temperature derating limit",
            "Check charger current setpoint over CAN and the derating logic/firmware")

    if masks["R6"].any():
        t0 = df.loc[masks["R6"], "time_s"].min()
        add("R6", t0, f"Controller temp up to {df['ctrl_temp_c'].max():.0f} C",
            "Motor controller overheating", "Check inverter cooling and sustained power demand")

    bad = [(cid, g, at) for cid, (g, at) in gaps.items() if g > 3.0]
    if bad:
        if len(bad) == len(CAN_IDS):
            add("R7", min(a for _, _, a in bad), "All CAN nodes lost messages", "Bus-level fault",
                "Check bus wiring, 120 ohm termination and common ground")
        for cid, g, at in bad:
            ok = [n for c, n in CAN_IDS.items() if c != cid and gaps[c][0] <= 3.0]
            add("R7", at, f"ID 0x{cid:03X} ({CAN_IDS[cid]}) silent for {g:.0f} s while {', '.join(ok)} kept transmitting",
                f"Node-specific fault: {CAN_IDS[cid]} ECU, its connector or CAN transceiver (bus-level fault unlikely)",
                "Check that ECU's power, ground, connector and transceiver")
    return out


# ------------------------------------------------------------------ 4. plots
def plot_run(df, masks, title, path):
    fig, ax = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    spec = [("pack_v", "Pack V"), ("pack_a", "Pack A"), ("batt_temp_c", "Batt temp C"),
            ("dcdc_v", "DC-DC V"), ("ctrl_temp_c", "Ctrl temp C")]
    bad_any = pd.concat(masks.values(), axis=1).fillna(False).any(axis=1)
    for a, (col, lab) in zip(ax, spec):
        a.plot(df["time_s"], df[col], lw=0.8)
        a.scatter(df.loc[bad_any, "time_s"], df.loc[bad_any, col], s=6, c="red")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].set_title(title + "  (red = limit violation)")
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_environment(data, path):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, df in data.items():
        ax[0].plot(df["time_s"], df["batt_temp_c"], label=name)
        ax[1].plot(df["time_s"], df["pack_v"], label=name, lw=0.7)
        ax[2].plot(df["time_s"], df["chg_a"], label=name)
    ax[0].set_ylabel("Battery temp (C)")
    ax[1].set_ylabel("Pack voltage (V)")
    ax[2].set_ylabel("Charger current (A)")
    ax[2].set_xlabel("Time (s)")
    ax[0].set_title("Environmental effect: cold / room / hot")
    ax[0].legend()
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ------------------------------------------------------------------ main
def main():
    runs = [
        ("Healthy -10C", -10, (), 1),
        ("Healthy 25C", 25, (), 2),
        ("Healthy 45C", 45, (), 3),
        ("Fault run A (25C)", 25, ("cooling", "dcdc", "can", "charger"), 4),
        ("Fault run B (25C)", 25, ("short",), 5),
    ]
    all_res, all_rca, logs, truths = [], [], [], {}
    decoded = {}
    for name, amb, faults, seed in runs:
        frames, truth = simulate(amb, faults, seed)
        df = decode(frames)
        res, masks, gaps = run_tests(name, df, frames)
        all_res += res
        all_rca += root_causes(name, df, masks, gaps)
        decoded[name] = df
        truths[name] = truth
        logs.append(df.assign(run=name))
        if faults:
            plot_run(df, masks, name, f"{name.split('(')[0].strip().replace(' ', '_').lower()}.png")

    plot_environment({k: v for k, v in decoded.items() if k.startswith("Healthy")}, "environment_comparison.png")

    # ---- ML layer: Isolation Forest on fault run A (unsupervised second opinion)
    dfA = decoded["Fault run A (25C)"]
    feat = dfA[["pack_v", "pack_a", "batt_temp_c", "dcdc_v", "ctrl_temp_c", "motor_rpm", "motor_torque_nm"]].interpolate().bfill().ffill()
    pred = IsolationForest(contamination=0.06, random_state=0).fit_predict(feat) == -1
    tA = truths["Fault run A (25C)"][dfA["time_s"].astype(int).values]
    inj = tA != "Normal"
    ml_hit, ml_fp = pred[inj].mean() * 100, pred[~inj].mean() * 100

    results = pd.DataFrame(all_res)
    matrix = results.pivot(index="run", columns="req", values="result").reindex([r[0] for r in runs])
    rca = pd.DataFrame(all_rca)
    summary = pd.DataFrame({
        "Metric": ["Test runs", "Requirements checked per run", "Requirements failed (all runs)",
                   "Root-cause findings", "Isolation Forest: injected-fault samples flagged (%)",
                   "Isolation Forest: normal samples flagged (%)"],
        "Value": [len(runs), len(REQ), int((results["result"] == "FAIL").sum()), len(rca),
                  round(ml_hit, 1), round(ml_fp, 1)],
    })

    out = "ev_powertrain_test_report.xlsx"
    with pd.ExcelWriter(out) as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        matrix.reset_index().to_excel(xw, sheet_name="Test Matrix", index=False)
        results.to_excel(xw, sheet_name="Test Results", index=False)
        rca.to_excel(xw, sheet_name="Root Cause Analysis", index=False)
        pd.concat(logs).to_excel(xw, sheet_name="Decoded CAN Logs", index=False)

    wb = load_workbook(out)
    green, red = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
    for sheet in ("Test Matrix", "Test Results"):
        ws = wb[sheet]
        for row in ws.iter_rows():
            for c in row:
                if c.value == "PASS":
                    c.fill = green
                elif c.value == "FAIL":
                    c.fill = red
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:60])
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 70)
    wb.save(out)

    print(matrix.to_string())
    print()
    print(rca[["run", "req", "time_s", "likely_root_cause"]].to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
