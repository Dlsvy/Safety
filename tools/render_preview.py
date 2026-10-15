"""결과 메쉬 미리보기 렌더 (Blender, 헤드리스).

blender -b -P tools/render_preview.py -- MESH.obj OUT.png [N_VIEWS] [ELEV_DEG]
텍스처를 발광(Emission)으로 그려 조명 영향 없이 사진 색 그대로 보여준다. 여러 방향을 한 장에 이어 붙인다.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, OUT = argv[0], argv[1]
NV = int(argv[2]) if len(argv) > 2 else 3
ELEV = float(argv[3]) if len(argv) > 3 else 25.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=MESH, forward_axis="Y", up_axis="Z")
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in objs:
    for slot in o.material_slots:
        m = slot.material
        if not m or not m.use_nodes:
            continue
        nt = m.node_tree
        tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        em = nt.nodes.new("ShaderNodeEmission")
        if tex:
            nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs[0])

# 정점 분포로 중심·반경 (이상치에 강하게 3~97% 범위, Blender 내장 파이썬엔 numpy 가 없을 수 있음)
pts = []
for o in objs:
    vs = o.data.vertices
    step = max(1, len(vs) // 50000)
    pts += [o.matrix_world @ vs[i].co for i in range(0, len(vs), step)]
lo, hi = [], []
for ax in range(3):
    v = sorted(p[ax] for p in pts)
    lo.append(v[int(0.03 * (len(v) - 1))])
    hi.append(v[int(0.97 * (len(v) - 1))])
c = Vector([(a + b) / 2 for a, b in zip(lo, hi)])
r = (Vector(hi) - Vector(lo)).length / 2

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 4
sc.cycles.use_denoising = False
sc.cycles.device = "CPU"
sc.render.resolution_x, sc.render.resolution_y = 960, 720
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.92, 0.93, 0.95, 1)
cd = bpy.data.cameras.new("c")
cd.lens = 30
cam = bpy.data.objects.new("c", cd)
sc.collection.objects.link(cam)
sc.camera = cam
files = []
for k in range(NV):
    az = math.radians(-90 + (k - (NV - 1) / 2) * 50)
    el = math.radians(ELEV)
    d = r * 1.6
    cam.location = c + Vector((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)))
    cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    f = OUT.replace(".png", f"_{k}.png")
    sc.render.filepath = f
    bpy.ops.render.render(write_still=True)
    files.append(f)
print("PREVIEWS", " ".join(files))
