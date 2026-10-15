"""가상 건설현장 렌더링 (정확도 검증용 정답 데이터 생성).

blender -b -P tools/synth_site_blender.py -- OUT_DIR N_IMAGES [WIDTH HEIGHT SAMPLES]

출력: OUT_DIR/render/*.png, gt_mesh.ply(Z-up, m), gt.json(카메라·마커 정답)
좌표: Z-up, m. 0번 마커 중심 = 원점, 마커 윗변 방향 = +X (photo3d 마커 모드와 같은 정의)
"""
import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
N = int(argv[1])
W = int(argv[2]) if len(argv) > 2 else 1600
H = int(argv[3]) if len(argv) > 3 else 1200
SPP = int(argv[4]) if len(argv) > 4 else 24
MARKER_DIR = os.path.join(OUT, "markers")
GT_ONLY = os.environ.get("GT_ONLY") == "1"  # 렌더 없이 정답 메쉬·gt.json 만 다시 쓰기
os.makedirs(os.path.join(OUT, "render"), exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = SPP
sc.cycles.use_denoising = False
sc.cycles.max_bounces = 3
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "Standard"

# ── 재질 ───────────────────────────────────────────────
def mat_proc(name, c1, c2, scale, kind="noise", brick=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 0.6  # 큰 얼룩(반복되지 않는 오염·습기 자국)
    n1.inputs["Detail"].default_value = 12.0
    n1.inputs["Roughness"].default_value = 0.65
    nt.links.new(mp.outputs["Vector"], n1.inputs["Vector"])
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.inputs["Scale"].default_value = 9.0
    nt.links.new(mp.outputs["Vector"], vo.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    nt.links.new(n1.outputs["Fac"], mix.inputs[0])
    mix.inputs[1].default_value = 0.7
    nt.links.new(vo.outputs["Distance"], mix.inputs[2])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(mix.outputs[0], ramp.inputs["Fac"])
    col = ramp.outputs["Color"]
    if brick:
        bt = nt.nodes.new("ShaderNodeTexBrick")
        bt.inputs["Scale"].default_value = brick
        bt.inputs["Brick Width"].default_value = 0.5
        bt.inputs["Row Height"].default_value = 0.25
        bt.inputs["Mortar Size"].default_value = 0.02
        bt.inputs["Color1"].default_value = (0.62, 0.28, 0.18, 1)
        bt.inputs["Color2"].default_value = (0.35, 0.16, 0.12, 1)
        bt.inputs["Bias"].default_value = 0.0  # 벽돌마다 색 무작위 혼합
        tc2 = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc2.outputs["Object"], sep.inputs[0])
        add = nt.nodes.new("ShaderNodeMath")
        nt.links.new(sep.outputs["X"], add.inputs[0])
        nt.links.new(sep.outputs["Y"], add.inputs[1])
        comb = nt.nodes.new("ShaderNodeCombineXYZ")  # 벽면 (수평거리, 높이) 로 벽돌 매핑
        nt.links.new(add.outputs[0], comb.inputs["X"])
        nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
        nt.links.new(comb.outputs[0], bt.inputs["Vector"])
        mx = nt.nodes.new("ShaderNodeMixRGB")
        mx.blend_type = "MULTIPLY"
        mx.inputs["Fac"].default_value = 1.0
        nt.links.new(bt.outputs["Color"], mx.inputs["Color1"])
        nt.links.new(col, mx.inputs["Color2"])
        col = mx.outputs["Color"]
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def mat_image(name, path):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Closest"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


M_GROUND = mat_proc("ground", (0.18, 0.16, 0.13), (0.62, 0.57, 0.48), 1.0)
M_CONC = mat_proc("concrete", (0.38, 0.38, 0.37), (0.80, 0.79, 0.76), 1.6)
M_BRICK = mat_proc("brick", (0.6, 0.6, 0.6), (1.0, 1.0, 1.0), 2.0, brick=1.2)
M_CONT = mat_proc("container", (0.05, 0.18, 0.45), (0.35, 0.55, 0.85), 2.5)
M_WOOD = mat_proc("wood", (0.30, 0.18, 0.08), (0.75, 0.55, 0.30), 4.0)
M_STEEL = mat_proc("steel", (0.25, 0.25, 0.27), (0.70, 0.62, 0.20), 5.0)

objs = []


def box(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    objs.append(o)
    return o


# 지면 80x80
bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
g = bpy.context.object
g.name = "ground"
g.data.materials.append(M_GROUND)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.subdivide(number_cuts=40)
bpy.ops.object.mode_set(mode="OBJECT")
objs.append(g)

# 공사 중 건물: 18 x 12 x 9 m, 중심 (4, 14)
BX, BY = 4.0, 14.0
box("building", (BX, BY, 4.5), (18, 12, 9), M_BRICK)
for k, z in enumerate((3.0, 6.0)):  # 층 슬래브 돌출(0.3m 캔틸레버)
    box(f"slab{k}", (BX, BY, z), (18.6, 12.6, 0.25), M_CONC)
box("roofslab", (BX, BY, 9.125), (18.6, 12.6, 0.25), M_CONC)
# 다음 층 기둥 (0.6 x 0.6 x 3 m) 4 x 3 개
for i in range(4):
    for j in range(3):
        box(f"col{i}{j}", (BX - 8.4 + i * 5.6, BY - 5.4 + j * 5.4, 9.25 + 1.5), (0.6, 0.6, 3.0), M_CONC)
# 20ft 컨테이너 6.06 x 2.44 x 2.59, 중심 (-12, 2)
box("container", (-12.0, 2.0, 1.295), (6.06, 2.44, 2.59), M_CONT)
# 자재 팔레트 더미
for k, (x, y, h) in enumerate([(9, -3, 1.0), (10.4, -3, 0.6), (9.7, -4.4, 1.4)]):
    box(f"pallet{k}", (x, y, h / 2), (1.2, 1.0, h), M_WOOD)
MARKERS = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (-8.0, 3.0), 3: (14.0, 10.0), 4: (-7.0, 16.0), 5: (4.0, 6.0)}
# 현장 잡동사니(자재·잔해·콘)— 실제 현장처럼 고유 특징을 제공
import random as _r
_r.seed(3)
for k in range(40):
    while True:
        x, y = _r.uniform(-22, 26), _r.uniform(-14, 30)
        near_marker = any((x - mx) ** 2 + (y - my) ** 2 < 1.5 ** 2 for mx, my in MARKERS.values())
        if not (-6.5 < x < 14.5 and 6.5 < y < 21.5) and not near_marker:
            break
    w, d, h = _r.uniform(0.2, 1.2), _r.uniform(0.2, 1.2), _r.uniform(0.1, 0.9)
    o = box(f"clutter{k}", (x, y, h / 2), (w, d, h), _r.choice([M_WOOD, M_CONT, M_CONC, M_STEEL]))
    o.rotation_euler[2] = _r.uniform(0, 3.14)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(rotation=True)

# 가로등 기둥 지름 0.3, 높이 6 (얇은 구조물 시험)
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.15, depth=6, location=(-6, -6, 3))
o = bpy.context.object
o.name = "pole"
o.data.materials.append(M_STEEL)
objs.append(o)

# ArUco 마커 (검은 테두리 한 변 0.5 m, 흰 여백 포함 0.7 m 판)
for mid, (x, y) in MARKERS.items():
    bpy.ops.mesh.primitive_plane_add(size=0.7, location=(x, y, 0.003))
    o = bpy.context.object
    o.name = f"marker{mid}"
    o.data.materials.append(mat_image(f"mk{mid}", os.path.join(MARKER_DIR, f"m{mid}.png")))
    objs.append(o)

# 조명: 하늘 + 태양
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.65, 0.85, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
sun = bpy.context.object
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(35))

# ── 카메라: 폰 26mm 환산, 4:3 ──────────────────────────────
cam_data = bpy.data.cameras.new("phone")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = 26.0 * (36.0 / 36.0)
cam = bpy.data.objects.new("phone", cam_data)
sc.collection.objects.link(cam)
sc.camera = cam

import random
random.seed(7)
poses = []
center = Vector((1.0, 8.0, 0.0))
for k in range(N):
    t = k / N * 2 * math.pi
    ring = k % 3  # 세 가지 높이/거리를 섞어 사람이 걸으며 찍는 상황을 흉내
    rx, ry = (24, 22) if ring == 0 else (20, 18) if ring == 1 else (30, 27)
    z = 1.5 if ring != 1 else 2.6
    p = center + Vector((rx * math.cos(t), ry * math.sin(t), 0)) + Vector(
        (random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7), z + random.uniform(-0.2, 0.2)))
    tgt = Vector((BX, BY, 4.0)) * 0.5 + center * 0.5 + Vector(
        (random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-1.0, 2.5)))
    if ring == 1:
        tgt.z = 0.0  # 낮게 내려다보며 마커·지면 촬영
    d = tgt - p
    cam.location = p
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(random.uniform(-3, 3)))
    bpy.context.view_layer.update()
    fn = f"IMG_{k:04d}.png"
    sc.render.filepath = os.path.join(OUT, "render", fn)
    if not GT_ONLY and not os.path.exists(sc.render.filepath):
        bpy.ops.render.render(write_still=True)
    poses.append({"name": fn, "location": list(cam.location),
                  "matrix_world": [list(r) for r in cam.matrix_world]})
    print(f"rendered {k + 1}/{N}", flush=True)

# 마커 근접 촬영 (가이드: 마커 쪽으로 다가가 아래를 향해 촬영) — 마커당 4방향, 5m 거리
CLOSE = int(os.environ.get("CLOSE_VIEWS", "4"))
k = len(poses)
for mid, (mx, my) in MARKERS.items():
    for q in range(CLOSE):
        a = 2 * math.pi * q / CLOSE + mid
        p = Vector((mx + 5 * math.cos(a), my + 5 * math.sin(a), 1.6 + random.uniform(-0.1, 0.3)))
        tgt = Vector((mx, my, 0)) + Vector((random.uniform(-1, 1), random.uniform(-1, 1), 1.0))
        cam.location = p
        cam.rotation_euler = (tgt - p).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        fn = f"IMG_{k:04d}.png"
        sc.render.filepath = os.path.join(OUT, "render", fn)
        if not GT_ONLY and not os.path.exists(sc.render.filepath):
            bpy.ops.render.render(write_still=True)
        poses.append({"name": fn, "location": list(cam.location),
                      "matrix_world": [list(r) for r in cam.matrix_world]})
        k += 1
        print(f"rendered close {k}", flush=True)

# 높은 위치 촬영 — 폴대 카메라(6 m)·드론(20 m). 지붕·윗면처럼 눈높이에서 안 보이는 면을 측정하기 위해
HIGH = int(os.environ.get("HIGH_VIEWS", "0"))
if HIGH:
    k = len(poses)
    shots = [(6.0, 26.0, 2.0) for _ in range(HIGH)] + [(20.0, 22.0, 0.0) for _ in range(HIGH // 2)]
    for q, (h, rad, tz) in enumerate(shots):
        n_ring = HIGH if h < 10 else HIGH // 2
        a = 2 * math.pi * (q % n_ring) / n_ring + (0.3 if h > 10 else 0)
        p = center + Vector((rad * math.cos(a), rad * math.sin(a), h))
        tgt = Vector((BX, BY, tz + 4.0)) * 0.6 + center * 0.4
        cam.location = p
        cam.rotation_euler = (tgt - p).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        fn = f"IMG_{k:04d}.png"
        sc.render.filepath = os.path.join(OUT, "render", fn)
        if not GT_ONLY and not os.path.exists(sc.render.filepath):
            bpy.ops.render.render(write_still=True)
        poses.append({"name": fn, "location": list(cam.location),
                      "matrix_world": [list(r) for r in cam.matrix_world]})
        k += 1
        print(f"rendered high {k}", flush=True)

# 정답 메쉬 (월드 좌표, 삼각형) — 물체마다 정점 번호를 이어 붙인다
verts, faces = [], []
for o in objs:
    if o.name.startswith("marker"):
        continue
    me = o.to_mesh()
    me.calc_loop_triangles()
    base = len(verts)
    verts += [o.matrix_world @ v.co for v in me.vertices]
    faces += [[base + i for i in t.vertices] for t in me.loop_triangles]
    o.to_mesh_clear()
with open(os.path.join(OUT, "gt_mesh.ply"), "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    for v in verts:
        f.write(f"{v.x:.5f} {v.y:.5f} {v.z:.5f}\n")
    for fc in faces:
        f.write(f"{len(fc)} " + " ".join(map(str, fc)) + "\n")

px = W / 36.0 * 26.0
json.dump({"image_size": [W, H], "focal_px": px, "markers": {k: [x, y, 0.003] for k, (x, y) in MARKERS.items()},
           "marker_size": 0.5, "poses": poses,
           "dims": {"building": [18, 12, 9], "container": [6.06, 2.44, 2.59], "pole_h": 6.0,
                    "column": [0.6, 0.6, 3.0]}},
          open(os.path.join(OUT, "gt.json"), "w"), indent=1)
print("DONE")
