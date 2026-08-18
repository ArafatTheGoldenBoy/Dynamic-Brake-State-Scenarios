# Dynamic Brake State Scenarios

Scenario-level braking experiments in [CARLA](https://carla.org/), from my M.Sc. thesis
on dynamic brake state estimation.

This repository holds the scenario driver — a single-file closed-loop stack
(`test6_motor.py`, ~530 lines) that runs an ego vehicle at 20 Hz with a front camera and
a top-down view, gates detections to the ego lane, estimates range from bounding-box
geometry, classifies traffic light colour in HSV, and maps a desired deceleration onto
throttle and brake.

> **Looking for the full thesis codebase?** The modular version — separate ECU,
> perception, planning, ABS, and telemetry modules — lives in
> **[Dynamic_brake_state_v2](https://github.com/ArafatTheGoldenBoy/Dynamic_brake_state_v2)**.
> This repository is the scenario-testing counterpart, kept separate because it is a
> self-contained script rather than a package.

## What `test6_motor.py` does

- **Closed-loop CARLA control** — synchronous mode at 20 Hz (`DT = 0.05`), pygame HUD
  showing the front camera and a top-down pane side by side.
- **Lane gating** — vehicles outside `LANE_HALF_WIDTH + LATERAL_MARGIN` (2.4 m) are
  ignored, so oncoming and adjacent-lane traffic does not trigger braking.
- **Range estimation** — monocular distance from bounding-box geometry using a focal
  length derived from the 90° horizontal FOV.
- **Traffic light state** — HSV masking with band-fraction analysis to read red, amber,
  and green rather than only detecting that a light exists.
- **Braking policy** — proportional throttle control (`KP_THROTTLE = 0.15`) toward a
  target speed, a comfort deceleration of 3.5 m/s² mapped onto brake in `[0, 1]`, a hard
  cap at 8.0 m/s², a 5 m safety distance with a 0.2 s reaction lag, and release timers
  for clear-ahead and stop-sign waits.
- **Surface friction** — `--mu` with `--apply-tire-friction` varies the friction
  coefficient, which is what makes the braking behaviour scenario-dependent.

## Requirements

- CARLA 0.9.x with its Python API on `PYTHONPATH`, or the `.egg` reachable at
  `../carla/dist/`
- Python 3.8+
- `numpy`, `opencv-python`, `pygame`

## Running

Start a CARLA server, then:

```bash
python test6_motor.py --host 127.0.0.1 --port 2000 --town Town10HD_Opt
```

Low-friction scenario:

```bash
python test6_motor.py --town Town10HD_Opt --mu 0.4 --apply-tire-friction
```

## Related repositories

| Repository | Role |
|---|---|
| [Dynamic_brake_state_v2](https://github.com/ArafatTheGoldenBoy/Dynamic_brake_state_v2) | Modular thesis codebase — ECU, perception, planning, ABS, telemetry |
| [Dynamic-Braking-System](https://github.com/ArafatTheGoldenBoy/Dynamic-Braking-System) | Earlier perception work: MobileNet-SSD to YOLOv10, traffic light training |
| [Vision-based-perception](https://github.com/ArafatTheGoldenBoy/Vision-based-perception) | Detector experiments (MobileNetV4-SSD) |

## License

MIT — see [LICENSE](LICENSE).
