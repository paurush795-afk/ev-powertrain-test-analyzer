# EV Powertrain CAN Test Analyzer with Environmental Testing and Root Cause Analysis

Python tool that simulates EV powertrain test runs, decodes raw CAN frames, checks them
against test requirements under cold / room / hot conditions, and produces an Excel test
report with root-cause analysis for every failure.

**All data is simulated (no real vehicle).** Test limits are illustrative engineering
limits, not clauses copied from a homologation standard.

## Systems covered
HV battery pack (voltage, current, temperature, SOC), HV fuse, DC-DC converter (LV output),
onboard charger (CC-CV with temperature derating), HV motor + motor controller.

## CAN protocol
Frames are encoded to raw bytes and decoded again with Python `struct` (scaled integers).
- 0x100 BMS: pack voltage, current, temperature, SOC, fuse state
- 0x200 DC-DC: output voltage, current, temperature
- 0x300 Charger: voltage, current, status
- 0x400 Motor controller: speed, torque, controller temperature

## Test requirements (R1-R7)
R1 pack voltage range | R2 battery temperature | R3 pack current and fuse state |
R4 DC-DC output range | R5 charger current vs temperature derating |
R6 motor controller temperature | R7 CAN message timeout per node

## Test runs
Healthy at -10 C, 25 C, 45 C (environmental effects: cold raises resistance and limits
charging current) and two fault-injection runs:
- A: cooling failure, DC-DC undervoltage, CAN timeout (motor controller), charger overcurrent
- B: HV short circuit that blows the HV fuse

## Root cause analysis
Rules combine evidence across signals, for example:
- Temperature rising fast at low current -> cooling failure, not load heating
- DC-DC low while HV pack voltage is normal -> DC-DC fault; DC-DC low with fuse open -> secondary effect
- One CAN node silent while others transmit -> node-specific fault, not bus-level

An Isolation Forest (scikit-learn) gives an unsupervised second opinion on fault run A.

## Run
pip install -r requirements.txt
python ev_powertrain_test_analyzer.py

## Output
ev_powertrain_test_report.xlsx (Summary, Test Matrix, Test Results, Root Cause Analysis,
Decoded CAN Logs), fault_run_a.png, fault_run_b.png, environment_comparison.png

## Results (from my run)
- 5 test runs x 7 requirements; healthy runs pass at -10 C, 25 C and 45 C
- All 5 injected faults detected; 6 root-cause findings generated
- Isolation Forest flagged 36.7% of injected-fault samples (3.4% false alarms), so the
  rule-based checks are the primary method and the ML layer is a secondary check

## Tech
Python, Pandas, NumPy, scikit-learn, Matplotlib, openpyxl, Excel, CAN, Git/GitHub
