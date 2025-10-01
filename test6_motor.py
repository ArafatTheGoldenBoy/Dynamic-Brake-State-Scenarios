from __future__ import print_function
import glob, os, sys, math, argparse, random, queue
from typing import Optional, Tuple
import numpy as np
import pygame
import cv2

# ===================== add CARLA egg to path =====================
try:
    sys.path.append(
        glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    )
except IndexError:
    pass

import carla  # after egg is on path


# ===================== configuration =====================
IMG_W, IMG_H      = 960, 540          # each pane (front + top)
FOV_X_DEG         = 90.0              # front cam horizontal FOV
DT                = 0.05              # 20 Hz (set 0.02 for lower latency)
FX                = (IMG_W / 2.0) / math.tan(math.radians(FOV_X_DEG / 2.0))

# Lane gating (ignore other-lane vehicles)
LANE_HALF_WIDTH   = 1.8
LATERAL_MARGIN    = 0.6
LATERAL_MAX       = LANE_HALF_WIDTH + LATERAL_MARGIN

# Cruise / braking
A_MAX             = 8.0               # hard cap
B_COMFORT         = 3.5               # map a_des -> [0..1] brake
V_TARGET          = 10.0              # m/s (36 km/h) default target
KP_THROTTLE       = 0.15
D_SAFETY          = 5.0
TAU               = 0.20
EPS               = 0.5
ALPHA_VBLEND      = 0.7               # blend velocity estimates
S_ENGAGE          = 35.0              # single threshold for non-TL
V_STOP            = 0.10
CLEAR_DELAY_S     = 3.0               # release timer when clear
STOP_WAIT_S       = 5.0               # stop-sign wait
KICK_SEC          = 0.6               # start “kick”
KICK_THR          = 0.25

# === MOTOR BRAKE / LOW-μ ===
MU_DEFAULT        = 0.90              # dry~0.9, wet~0.6, ice~0.2
REV_PULSE_V_MAX   = 2.0               # reverse torque below this speed
REV_THR           = 0.18              # reverse torque pulse
ABS_V_MAX         = 4.0               # micro-ABS below this speed
ABS_B_MIN         = 0.20              # only PWM brake if above this
ABS_PWM_SCALE     = 0.5               # halve brake every other tick

# Detection
CONF_THR_DEFAULT  = 0.40
NMS_THR           = 0.45
H_MIN_PX          = 16
CENTER_BAND_FRAC  = 0.35

# Traffic light ROI + logic
TL_IOU_THRESH     = 0.30              # (not used in this simplified release)
S_TL_ENGAGE       = 55.0
TL_ROI_YMAX_FRAC  = 0.70
TL_ROI_XCENTER_FRAC = 0.50

# Classes
VEHICLE_CLASSES     = {'car','bus','truck','motorcycle','motorbike','bicycle','train'}
PEDESTRIAN_CLASSES  = {'person'}
TRIGGER_CLASSES     = VEHICLE_CLASSES | {'traffic light','stop sign'} | PEDESTRIAN_CLASSES

# Approx real heights (meters) for monocular pinhole
OBJ_HEIGHT_M = {
    'person': 1.70,
    'car': 1.50,
    'traffic light': 2.20,
    'bus': 3.00,
    'truck': 3.20,
    'motorcycle': 1.40,
    'motorbike': 1.40,
    'bicycle': 1.40,
    'train': 3.50,
    'stop sign': 0.75,
}

# Debug toggle for traffic light mask visualization
DEBUG_TL = True

# ---------- label normalization ----------
def _norm_label(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalpha())  # 'traffic light' -> 'trafficlight'

TRIGGER_NAMES_NORM = {
    'trafficlight','stopsign','person','car','bus','truck','motorcycle','motorbike','bicycle','train'
}

# ===================== helpers =====================
def fov_y_from_x(width: int, height: int, fov_x_deg: float) -> float:
    fov_x = math.radians(fov_x_deg)
    return 2.0 * math.atan((height / width) * math.tan(fov_x / 2.0))

def focal_length_y_px(width: int, height: int, fov_x_deg: float) -> float:
    fovy = fov_y_from_x(width, height, fov_x_deg)
    return (height / 2.0) / math.tan(fovy / 2.0)

def bgr_to_pygame_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

def carla_image_to_surface(image) -> pygame.Surface:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

# Robust traffic light color detection (with debug overlay)
def detect_tl_color(roi_bgr: np.ndarray) -> str:
    if roi_bgr is None or roi_bgr.size == 0:
        return 'Unknown'
    h, w = roi_bgr.shape[:2]
    if h < 8 or w < 8:
        return 'Unknown'
    pad = int(0.15 * min(h, w))
    y0, y1 = pad, max(pad + 1, h - pad)
    x0, x1 = pad, max(pad + 1, w - pad)
    roi = roi_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = roi_bgr
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S_MIN, V_MIN = 110, 150
    def mask_range(lo_h, hi_h):
        lo = (lo_h, S_MIN, V_MIN); hi = (hi_h, 255, 255)
        return cv2.inRange(hsv, lo, hi)
    red  = cv2.bitwise_or(mask_range(0,10), mask_range(170,180))
    yellow = mask_range(20,33)   # tighter to avoid housings
    green  = mask_range(45,85)
    k = np.ones((3,3), np.uint8)
    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.dilate(m, k, iterations=1)
        return m
    red, yellow, green = clean(red), clean(yellow), clean(green)
    if DEBUG_TL:
        dbg = cv2.merge([green, yellow, red])
        small = cv2.resize(dbg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        h0, w0 = small.shape[:2]; y_off, x_off = 2, 2
        y1 = min(roi_bgr.shape[0], y_off + h0); x1 = min(roi_bgr.shape[1], x_off + w0)
        roi_bgr[y_off:y1, x_off:x1] = small[:y1 - y_off, :x1 - x_off]
    def largest_area(m):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        return 0 if n <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
    areas = {'Red':largest_area(red),'Yellow':largest_area(yellow),'Green':largest_area(green)}
    H = roi.shape[0]; thirds = [(0,H//3),(H//3,2*H//3),(2*H//3,H)]
    def frac_in_band(m, band_idx):
        a,b = thirds[band_idx]; tot = cv2.countNonZero(m)
        if tot == 0: return 0.0
        sub = cv2.countNonZero(m[a:b,:]); return sub/float(tot)
    priors = {'Red':0.7+0.6*frac_in_band(red,0),'Yellow':0.7+0.6*frac_in_band(yellow,1),'Green':0.7+0.6*frac_in_band(green,2)}
    scores = {c: areas[c]*priors[c] for c in areas}
    best = max(scores, key=scores.get)
    min_pixels = max(60, int(0.002*(roi.shape[0]*roi.shape[1])))
    return best if areas[best] >= min_pixels else 'Unknown'

def wrap_pi(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def iou_xywh(a: Optional[Tuple[int,int,int,int]], b: Optional[Tuple[int,int,int,int]]) -> float:
    if not a or not b: return 0.0
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax+aw, ay+ah, bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a, area_b = aw*ah, bw*bh
    return inter / float(area_a + area_b - inter)

def shadow(surface: pygame.Surface, text: str, pos, color, shadow_color=(0,0,0), offset=1):
    font = pygame.font.SysFont('Arial', 20)
    s = font.render(text, True, shadow_color); surface.blit(s, (pos[0]+offset, pos[1]+offset))
    s2 = font.render(text, True, color);        surface.blit(s2, pos)

def _fallback_labels_91():
    # 91-entry list (index = classId, we will use [cid-1] so make it 90; fill essentials)
    # OpenCV's COCO model uses 1-based ids. We'll build a 90-length list.
    L = [""]*90
    # Populate essentials by COCO ids
    mapping = {
        1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 6:"bus", 7:"train", 8:"truck",
        10:"traffic light", 13:"stop sign"
    }
    for k,v in mapping.items():
        idx = k-1
        if 0 <= idx < 90:
            L[idx] = v
    return L

# ===================== main =====================
def main():
    parser = argparse.ArgumentParser(description='Nearest-first + TL/StopSign (robust TL color + motor brake + odometer)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default=None, help='Town name, e.g., Town03 or Town05_Opt')
    parser.add_argument('--mu', type=float, default=MU_DEFAULT, help='Road friction estimate (dry~0.9, wet~0.6, ice~0.2)')
    parser.add_argument('--apply-tire-friction', action='store_true',
                        help='Also set wheel.tire_friction≈mu to make the sim physically slick.')
    args = parser.parse_args()

    pygame.init()
    WIN_W, WIN_H = IMG_W * 2, IMG_H
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Nearest-first + TL/StopSign | Sync')
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port); client.set_timeout(10.0)
    if args.town:
        try:
            w0 = client.get_world(); s0 = w0.get_settings()
            if s0.synchronous_mode:
                s0.synchronous_mode = False; w0.apply_settings(s0)
        except Exception:
            pass
        world = client.load_world(args.town)
    else:
        world = client.get_world()

    original_settings = world.get_settings()
    tm = client.get_trafficmanager()
    carla_map = world.get_map()

    # Detector (OpenCV DNN SSD) with graceful fallback if files are missing
    cfg = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    mdl = 'frozen_inference_graph.pb'
    DETECT_ENABLED = os.path.exists(cfg) and os.path.exists(mdl)
    labels = None
    try:
        with open('labels.txt', 'rt') as f:
            labels = f.read().strip().splitlines()
    except Exception:
        labels = _fallback_labels_91()

    if DETECT_ENABLED:
        net = cv2.dnn_DetectionModel(mdl, cfg)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
    else:
        print("[WARN] SSD model files not found. Running without perception triggers.")

    FY = focal_length_y_px(IMG_W, IMG_H, FOV_X_DEG)

    actors_to_destroy = []
    hold_blocked = False
    hold_reason   = None
    last_s0 = None
    prev_loc = None
    sim_time = 0.0
    kick_until = 0.0
    stop_latch_time = -1.0
    last_tl_red_box: Optional[Tuple[int,int,int,int]] = None
    no_trigger_elapsed = 0.0
    no_red_elapsed = 0.0
    hud_msg = ''
    hud_until = 0.0
    conf_thr = CONF_THR_DEFAULT
    MU = max(0.05, min(1.2, args.mu))
    A_MU = MU * 9.81

    # live-adjustable target speed (m/s)
    v_target = float(V_TARGET)

    # odometer (meters)
    dist_total = 0.0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawns = world.get_map().get_spawn_points()
        ego = world.try_spawn_actor(ego_bp, random.choice(spawns) if spawns else carla.Transform())
        if ego is None:
            ego = world.spawn_actor(ego_bp, carla.Transform())
        actors_to_destroy.append(ego)
        ego.set_autopilot(False)

        # Physics headroom & (optional) tire friction ≈ μ
        phys = ego.get_physics_control()
        for w in phys.wheels:
            w.max_brake_torque     = max(8000.0, getattr(w, 'max_brake_torque', 4000.0))
            w.max_handbrake_torque = max(12000.0, getattr(w, 'max_handbrake_torque', 8000.0))
            if args.apply_tire_friction:
                w.tire_friction = MU
        ego.apply_physics_control(phys)

        # Cameras
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMG_W))
        cam_bp.set_attribute('image_size_y', str(IMG_H))
        cam_bp.set_attribute('fov', str(FOV_X_DEG))
        front_rel = carla.Transform(carla.Location(x=1.6, z=1.5))
        cam_front = world.spawn_actor(cam_bp, front_rel, attach_to=ego); actors_to_destroy.append(cam_front)
        top_rel = carla.Transform(carla.Location(x=0.0, z=25.0), carla.Rotation(pitch=-90.0))
        cam_top = world.spawn_actor(cam_bp, top_rel, attach_to=ego); actors_to_destroy.append(cam_top)

        q_front, q_top = queue.Queue(), queue.Queue()
        cam_front.listen(q_front.put); cam_top.listen(q_top.put)

        running = True
        while running:
            frame_id = world.tick()
            sim_time += DT

            img_front = q_front.get(timeout=2.0)
            img_top   = q_top.get(timeout=2.0)

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_LEFTBRACKET:
                        conf_thr = max(0.05, round(conf_thr - 0.05, 2))
                        hud_msg = f'conf -> {conf_thr:.2f}'; hud_until = sim_time + 2.0
                    elif e.key == pygame.K_RIGHTBRACKET:
                        conf_thr = min(0.99, round(conf_thr + 0.05, 2))
                        hud_msg = f'conf -> {conf_thr:.2f}'; hud_until = sim_time + 2.0
                    elif e.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        v_target = min(40.0, v_target + 0.5556)  # +2 km/h
                        hud_msg = f'Vtgt -> {v_target*3.6:.0f} km/h'; hud_until = sim_time + 2.0
                    elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        v_target = max(2.0, v_target - 0.5556)   # -2 km/h
                        hud_msg = f'Vtgt -> {v_target*3.6:.0f} km/h'; hud_until = sim_time + 2.0
                    elif e.key == pygame.K_0:
                        v_target = float(V_TARGET)
                        hud_msg = f'Vtgt reset -> {V_TARGET*3.6:.0f} km/h'; hud_until = sim_time + 2.0

            # Ego kinematics
            tr = ego.get_transform()
            loc = tr.location
            vel = ego.get_velocity()
            v_raw = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            if prev_loc is None:
                v_disp = 0.0
            else:
                dx = loc.x - prev_loc.x; dy = loc.y - prev_loc.y; dz = loc.z - prev_loc.z
                step = math.sqrt(dx*dx + dy*dy + dz*dz)
                v_disp = step / DT
                dist_total += step
            prev_loc = loc
            v = ALPHA_VBLEND * v_raw + (1.0 - ALPHA_VBLEND) * v_disp
            if v < 0.05: v = 0.0

            # Make a writable BGR
            arr = np.frombuffer(img_front.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
            bgr = arr[:, :, :3].copy()

            # Detection
            nearest_s = None
            nearest_kind = None
            nearest_box = None
            nearest_thr = None
            any_red_tl = False

            if DETECT_ENABLED:
                classIds, confs, boxes = net.detect(bgr, confThreshold=conf_thr, nmsThreshold=NMS_THR)
            else:
                classIds, confs, boxes = [], [], []

            if len(classIds) != 0:
                cx0 = IMG_W/2.0
                band_px = CENTER_BAND_FRAC * IMG_W
                for cid, conf, box in zip(np.array(classIds).flatten(), np.array(confs).flatten(), boxes):
                    x, y, w, h = map(int, box)
                    name = labels[cid - 1] if 0 <= cid - 1 < len(labels) and labels[cid - 1] else str(cid)
                    norm = _norm_label(name)

                    # coarse filter (allow 'traffic_light' variants)
                    if (norm not in TRIGGER_NAMES_NORM) or (h < H_MIN_PX):
                        if not (('traffic' in norm and 'light' in norm) and h >= H_MIN_PX):
                            continue

                    xc = x + w/2.0

                    # distance estimate via pinhole
                    H_real = OBJ_HEIGHT_M.get(name, OBJ_HEIGHT_M.get('traffic light') if ('traffic' in norm and 'light' in norm) else None)
                    if H_real is None:
                        continue
                    s0 = (FY * H_real) / float(h)

                    # lane gating for vehicles (keep pedestrians regardless)
                    if norm in VEHICLE_CLASSES:
                        if abs(xc - cx0) > band_px:
                            continue
                        lateral = ((xc - cx0) / max(1e-6, FX)) * max(1.0, s0)
                        if abs(lateral) > LATERAL_MAX:
                            continue

                    kind = None
                    thr_for_kind = None  # default (no engagement)

                    # Traffic light logic (wide ROI)
                    if norm == 'trafficlight' or ('traffic' in norm and 'light' in norm):
                        tl_center_ok = ((y + h/2.0) <= (TL_ROI_YMAX_FRAC * IMG_H) and
                                        (abs((x + w/2.0) - (IMG_W/2.0)) <= TL_ROI_XCENTER_FRAC * IMG_W))
                        tlc = 'Unknown'
                        if tl_center_ok:
                            roi = bgr[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
                            tlc = detect_tl_color(roi)
                        cv2.putText(bgr, f'TL:{tlc}', (x, y+min(h-2,16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0,255,0) if tlc=='Green' else (0,0,255) if tlc=='Red' else (200,200,0), 1)
                        if tlc == 'Red':
                            any_red_tl = True
                            last_tl_red_box = (x, y, w, h)
                            kind = 'traffic light (Red)'
                            thr_for_kind = S_TL_ENGAGE
                        else:
                            kind = f'traffic light ({tlc})'
                            thr_for_kind = None  # do not engage on non-red

                    elif norm == 'stopsign':
                        kind = 'stop sign'
                        thr_for_kind = S_ENGAGE

                    elif (norm in VEHICLE_CLASSES) or (norm in PEDESTRIAN_CLASSES):
                        kind = name  # keep raw for HUD
                        thr_for_kind = S_ENGAGE

                    # draw
                    color = (0,255,255)
                    if norm == 'stopsign':
                        color = (0,0,255)
                    cv2.rectangle(bgr, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(bgr, f'{name} {s0:.1f}m', (x, max(20, y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if kind is None:
                        continue

                    # nearest-first: update if strictly nearer
                    if (nearest_s is None) or (s0 < nearest_s):
                        nearest_s = s0
                        nearest_kind = kind
                        nearest_box = (x, y, w, h)
                        nearest_thr = thr_for_kind

            # timers
            if nearest_s is None:
                no_trigger_elapsed += DT
            else:
                no_trigger_elapsed = 0.0

            if any_red_tl:
                no_red_elapsed = 0.0
            else:
                no_red_elapsed += DT

            trigger_name = nearest_kind
            if nearest_s is not None:
                last_s0 = nearest_s

            # --- controller + latch ---
            throttle = 0.0
            brake = 0.0
            in_brake_band = (nearest_s is not None) and (nearest_thr is not None) and (nearest_s <= nearest_thr)
            ctrl = None  # may build a special control for motor-brake path

            if in_brake_band:
                s_used = 0.7 * (last_s0 if last_s0 is not None else nearest_s) + 0.3 * nearest_s
                s_eff  = max(s_used - D_SAFETY - v*TAU, EPS)
                a_des  = min((v*v) / (2.0 * s_eff), A_MAX)

                # μ·g clamp
                a_des  = min(a_des, A_MU)

                # map to brake command
                brake  = max(0.0, min(1.0, a_des / B_COMFORT))

                # micro-ABS at low speed
                if v < ABS_V_MAX and brake > ABS_B_MIN and ((frame_id % 2) == 0):
                    brake *= ABS_PWM_SCALE

                # reverse-torque pulse near stop
                use_motor_brake = (v < REV_PULSE_V_MAX) and (a_des > 0.0)
                if use_motor_brake:
                    ctrl = carla.VehicleControl(
                        throttle=REV_THR, brake=0.0, reverse=True,
                        steer=0.0, hand_brake=False
                    )

                if v < V_STOP:
                    hold_blocked = True
                    if trigger_name == 'traffic light (Red)':
                        hold_reason = 'red_light'
                    elif trigger_name and 'stop sign' in trigger_name:
                        hold_reason = 'stop_sign'
                        stop_latch_time = sim_time
                    else:
                        hold_reason = 'obstacle'

            elif hold_blocked:
                release = False
                if hold_reason == 'red_light' and no_red_elapsed >= CLEAR_DELAY_S:
                    release = True
                elif hold_reason == 'stop_sign':
                    if (sim_time - stop_latch_time) >= STOP_WAIT_S:
                        release = True
                elif hold_reason == 'obstacle' and no_trigger_elapsed >= CLEAR_DELAY_S:
                    release = True

                if release:
                    hold_blocked = False
                    hold_reason  = None
                    last_s0      = None
                    throttle, brake = 0.0, 0.0
                    kick_until = sim_time + KICK_SEC
                else:
                    throttle, brake = 0.0, 1.0

            else:
                # Cruise
                e_v = v_target - v
                throttle = max(0.0, min(1.0, KP_THROTTLE * e_v))
                brake = 0.0
                if sim_time < kick_until and v < 0.3:
                    throttle = max(throttle, KICK_THR)
                # ensure we get rolling from standstill
                if not hold_blocked and v < 0.25:
                    throttle = max(throttle, 0.35)

            # --- simple lane-follow steering ---
            tr = ego.get_transform()
            loc = tr.location
            yaw = math.radians(tr.rotation.yaw)
            wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            LA = max(6.0, min(12.0, 0.8 * max(v, 1.0)))
            next_wps = wp.next(LA) or wp.next(5.0)
            steer = 0.0
            if next_wps:
                best, best_diff = None, 1e9
                for cand in next_wps:
                    yaw_t = math.radians(cand.transform.rotation.yaw)
                    diff = abs(wrap_pi(yaw_t - yaw))
                    if diff < best_diff:
                        best, best_diff = cand, diff
                tx, ty = best.transform.location.x, best.transform.location.y
                dx, dy = tx - loc.x, ty - loc.y
                angle_to_point = math.atan2(dy, dx)
                heading_error  = wrap_pi(angle_to_point - yaw)
                cross_track    = (-math.sin(yaw))*dx + (math.cos(yaw))*dy
                steer_cmd = heading_error + math.atan2(0.8 * cross_track, v + 1e-3)
                steer = max(-1.0, min(1.0, steer_cmd))

            # Apply control (merge steer into whichever path we’re on)
            if in_brake_band and ctrl is not None:
                ctrl.steer = steer
                ego.apply_control(ctrl)
            else:
                ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, hand_brake=False))

            # --- render
            surf_front = bgr_to_pygame_surface(bgr)
            surf_top   = carla_image_to_surface(img_top)
            screen.blit(surf_front, (0, 0))
            screen.blit(surf_top,   (IMG_W, 0))

            # HUD
            v_kmh = v * 3.6
            txt1 = f'Frame {frame_id} | v={v_kmh:5.1f} km/h | trigger={trigger_name or "None"}'
            txt2 = f'thr={throttle:.2f}  brk={brake:.2f}  hold={hold_blocked}({hold_reason})  clear={no_trigger_elapsed:.1f}s  red_clear={no_red_elapsed:.1f}s'
            txt3 = f'conf={conf_thr:.2f} nms={NMS_THR:.2f} tl_iou={TL_IOU_THRESH:.2f}  mu={MU:.2f}  vtgt={v_target*3.6:.0f} km/h  dist={dist_total:6.1f} m'
            shadow(screen, txt3, (10, IMG_H-66), (160,220,255))
            shadow(screen, txt1, (10, IMG_H-44), (255,255,255))
            shadow(screen, txt2, (10, IMG_H-22), (0,255,160))
            if hud_msg and sim_time < hud_until:
                shadow(screen, hud_msg, (10, 10), (255,255,0))

            pygame.display.flip()
            clock.tick(60)

    finally:
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original_settings)
        for a in actors_to_destroy[::-1]:
            try:
                if 'sensor' in a.type_id:
                    a.stop()
                a.destroy()
            except Exception:
                pass
        pygame.quit()
        print('Shutdown cleanly.')


if __name__ == '__main__':
    main()
