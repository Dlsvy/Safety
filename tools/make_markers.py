"""현장용 ArUco 마커 생성.

python tools/make_markers.py OUT_DIR --ids 0-9 --size-cm 25 --paper A3
 → OUT_DIR/markers_A3.pdf (실측 크기, 100% 배율로 인쇄), OUT_DIR/m{id}.png (텍스처용)

인쇄 후 반드시 검은 테두리 한 변을 자로 재서 --marker-size 에 그 값을 넣으세요.
권장: 작업 반경 20~30m 현장은 A3에 25cm, 50m 이상은 A2/A1에 40~60cm.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PAPER = {"A4": (21.0, 29.7), "A3": (29.7, 42.0), "A2": (42.0, 59.4), "A1": (59.4, 84.1)}
DICTS = {"4x4_50": cv2.aruco.DICT_4X4_50, "5x5_100": cv2.aruco.DICT_5X5_100}


def ids_arg(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ids", type=ids_arg, default=ids_arg("0-9"))
    ap.add_argument("--size-cm", type=float, default=25.0)
    ap.add_argument("--paper", default="A3", choices=list(PAPER))
    ap.add_argument("--dict", default="4x4_50", choices=list(DICTS))
    ap.add_argument("--texture-border", type=float, default=0.2,
                    help="텍스처 PNG의 흰 여백(마커 한 변 대비 비율, 한쪽)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(DICTS[a.dict])
    dpi = 300
    pw, ph = PAPER[a.paper]
    pages = []
    for i in a.ids:
        # 텍스처용 PNG (마커 600px + 흰 여백)
        m = cv2.aruco.generateImageMarker(d, i, 600)
        b = int(600 * a.texture_border)
        cv2.imwrite(str(out / f"m{i}.png"), cv2.copyMakeBorder(m, b, b, b, b, cv2.BORDER_CONSTANT, value=255))
        # 인쇄용 페이지
        W, H = int(pw / 2.54 * dpi), int(ph / 2.54 * dpi)
        page = Image.new("L", (W, H), 255)
        s = int(round(a.size_cm / 2.54 * dpi))
        mk = Image.fromarray(cv2.aruco.generateImageMarker(d, i, s))
        x0, y0 = (W - s) // 2, (H - s) // 2
        page.paste(mk, (x0, y0))
        dr = ImageDraw.Draw(page)
        dr.text((x0, y0 + s + 40), f"ID {i}  |  {a.dict}  |  black edge = {a.size_cm:.1f} cm  |  print at 100%",
                fill=0)
        dr.line([(x0, y0 - 30), (x0 + s, y0 - 30)], fill=0, width=3)  # 실측 확인용 눈금선
        pages.append(page)
    pages[0].save(out / f"markers_{a.paper}.pdf", save_all=True, append_images=pages[1:], resolution=dpi)
    print(f"{len(pages)}장 → {out}")


if __name__ == "__main__":
    main()
