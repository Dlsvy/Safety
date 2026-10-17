"""PyTorch 2.6+ 윈도우 버그 우회: nvcc 로 .cu 를 빌드하면 torch 헤더에서 "'std': 모호한 기호"(C2872).
컴파일 오류가 난 헤더(기본: dynamo/compiled_autograd.h)의 std:: 를 ::std:: 로 바꾼다(뜻은 같고 전역 std 를 콕 집음).
사용: <venv>\\Scripts\\python.exe torch_win_fix.py [추가 헤더 상대경로 ...]
"""
import pathlib
import re
import sys

import torch

inc = pathlib.Path(torch.__file__).parent / "include"
targets = ["torch/csrc/dynamo/compiled_autograd.h"] + sys.argv[1:]
print(f"torch {torch.__version__}")
for rel in targets:
    p = inc / rel
    if not p.exists():
        print(f"  {rel}: 없음(이 torch 버전엔 해당 없음)")
        continue
    b = p.read_bytes()
    s = b.decode("utf-8", errors="surrogateescape")  # 인코딩이 이상해도 바이트 그대로 보존
    t = re.sub(r"(?<![\w:])std::", "::std::", s)
    # 특수화 선언(template<> struct std::hash<..>)은 :: 를 붙이면 문법 오류 → 원래대로(예전 패치본도 고침)
    t = re.sub(r"\b(struct|class)\s+::std::", r"\1 std::", t)
    if t == s:
        print(f"  {rel}: 이미 패치됨")
    else:
        p.write_bytes(t.encode("utf-8", errors="surrogateescape"))
        print(f"  {rel}: 패치함 ({s.count('std::') - s.count('::std::')}곳)")
