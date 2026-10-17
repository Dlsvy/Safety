"""MeshSplatting CUDA 코드의 리눅스 전용 문법을 윈도우(MSVC + nvcc)에서도 컴파일되게 고친다(한 번만, 뜻은 같음).
  forward.cu: `or` 키워드(MSVC 미지원) → ||, `uint` 형(리눅스 헤더에만 있음) → unsigned int
  utils.cu  : 장치 코드의 sqrt(int)·pow(int,int) → 윈도우에선 호스트 함수로 잡힘 → sqrtf·powf(float)
사용: python meshsplat_win_fix.py <mesh-splatting 폴더>
"""
import pathlib
import sys

src = pathlib.Path(sys.argv[1]) / "submodules" / "diff-triangle-mesh-rasterization" / "cuda_rasterizer"
FIXES = {
    "forward.cu": [
        ("distance_points > 1600 or distance_points < 1 or dist > -1",
         "distance_points > 1600 || distance_points < 1 || dist > -1"),
        ("uint bx_min = (uint)", "unsigned int bx_min = (unsigned int)"),
        ("uint by_min = (uint)", "unsigned int by_min = (unsigned int)"),
        ("uint bx_max = (uint)", "unsigned int bx_max = (unsigned int)"),
        ("uint by_max = (uint)", "unsigned int by_max = (unsigned int)"),
    ],
    "utils.cu": [
        ("(pow(-1, k) / sqrt(k + 1)) * pow(opacity_new[idx], k + 1)",
         "(((k & 1) ? -1.0f : 1.0f) / sqrtf((float)(k + 1))) * powf(opacity_new[idx], (float)(k + 1))"),
    ],
}
for name, reps in FIXES.items():
    p = src / name
    s = p.read_text(encoding="utf-8", errors="surrogateescape")
    n = 0
    for a, b in reps:
        if a in s:
            s = s.replace(a, b)
            n += 1
    if n:
        p.write_text(s, encoding="utf-8", errors="surrogateescape")
    print(f"  {name}: {'고침 ' + str(n) + '곳' if n else '이미 고쳐짐(또는 해당 없음)'}")
