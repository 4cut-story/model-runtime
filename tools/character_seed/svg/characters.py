"""Claude가 직접 그리는 캐릭터 참조 그림 (SVG). GPU 없이 만든다.

  python characters.py out_svg/   → out_svg/<THEME>/<이름>.svg

얼굴이 정면으로 크게 보이게 상반신만 그린다 (IP-Adapter Plus-Face 참조용).
테마마다 선 굵기·색, 배경, 질감 필터를 바꿔 conversation-service Concept.style과 그림체를 맞춘다.
"""

from __future__ import annotations

import sys
from pathlib import Path

W = 768
CX, CY, RX, RY = 384, 300, 118, 132  # 머리

COLORS = {
    "black": "#26221f", "brown": "#7a4b2a", "dark brown": "#4a2e1c", "auburn": "#a0442a",
    "blonde": "#e0b85c", "gray": "#8a8a8a",
    "yellow": "#f4c430", "green": "#5f9e5a", "navy": "#2f3e6b", "red": "#c8413b",
    "white": "#f7f5ef", "sky blue": "#86c5ea", "olive": "#7b7d3a", "blue": "#3d6fb6",
    "pink": "#ef9db4", "purple": "#8e6bb8", "orange": "#ec8a3a", "teal": "#2f9e95",
    "cream": "#f3e7c9", "tan": "#c9a46a", "mint": "#9fd8c4", "jade": "#7fb8a0", "indigo": "#3b3f7a", "crimson": "#a3242f",
}

SKIN = {"light": "#f8dcc4", "warm": "#f1c9a5", "tan": "#ddb08a"}

THEMES = {
    "DAILY": dict(stroke="#222", sw=7, bg="flat", bgc="#eef3f7", filt=None),
    "ANIMALS": dict(stroke="#6b5853", sw=5, bg="dots", bgc="#fdf1e6", filt=None),
    "WATERCOLOR": dict(stroke="#6e5446", sw=3, bg="wash", bgc="#f6efe2", filt="soft"),
    "PICTURE_BOOK": dict(stroke="#3a3330", sw=5, bg="paper", bgc="#fffdf8", filt="crayon"),
    "NEWSPAPER": dict(stroke="#111", sw=5, bg="halftone", bgc="#ffffff", filt=None, gray=True),
    "JOSEON": dict(stroke="#2b2522", sw=4, bg="hanji", bgc="#efe2c4", filt="brush", wash=True),
    "WUXIA": dict(stroke="#141414", sw=6, bg="mountains", bgc="#dfe4e3", filt="brush"),
}


def full(hexc: str) -> str:
    return "#" + "".join(ch * 2 for ch in hexc[1:]) if len(hexc) == 4 else hexc


def gray(hexc: str) -> str:
    hexc = full(hexc)
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    v = int(0.3 * r + 0.59 * g + 0.11 * b)
    v = 40 if v < 70 else (250 if v > 225 else v)
    return f"#{v:02x}{v:02x}{v:02x}"


def wash(hexc: str) -> str:
    hexc = full(hexc)
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    m = (r + g + b) / 3
    r, g, b = (int(0.75 * c + 0.25 * m) for c in (r, g, b))  # 채도를 조금 낮춘다
    return f"#{r:02x}{g:02x}{b:02x}"


class Pen:
    def __init__(self, theme: str) -> None:
        self.t = THEMES[theme]
        self.theme = theme

    def c(self, name: str) -> str:
        h = COLORS.get(name, name)
        if self.t.get("gray"):
            return gray(h)
        if self.t.get("wash"):
            return wash(h)
        return h

    def st(self, w: float | None = None) -> str:
        return f'stroke="{self.t["stroke"]}" stroke-width="{w or self.t["sw"]}" stroke-linejoin="round" stroke-linecap="round"'


def defs(p: Pen) -> str:
    out = ['<defs>']
    out.append('<filter id="crayon"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3"/>'
               '<feDisplacementMap in="SourceGraphic" scale="5"/></filter>')
    out.append('<filter id="brush"><feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="2" seed="7"/>'
               '<feDisplacementMap in="SourceGraphic" scale="4"/></filter>')
    out.append('<filter id="soft"><feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="3" seed="11"/>'
               '<feDisplacementMap in="SourceGraphic" scale="3"/></filter>')
    out.append('<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" seed="5"/>'
               '<feColorMatrix values="0 0 0 0 0.45  0 0 0 0 0.36  0 0 0 0 0.22  0 0 0 0.18 0"/></filter>')
    out.append('<filter id="blur30"><feGaussianBlur stdDeviation="30"/></filter>')
    out.append('<pattern id="dots" width="10" height="10" patternUnits="userSpaceOnUse">'
               '<circle cx="5" cy="5" r="1.6" fill="#bbb"/></pattern>')
    out.append('<pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               '<line x1="0" y1="0" x2="0" y2="9" stroke="#111" stroke-width="2"/></pattern>')
    out.append('<pattern id="plaid" width="40" height="40" patternUnits="userSpaceOnUse">'
               '<rect width="40" height="40" fill="none"/><rect x="0" y="14" width="40" height="8" fill="#000" opacity="0.22"/>'
               '<rect x="14" y="0" width="8" height="40" fill="#000" opacity="0.22"/>'
               '<rect x="0" y="30" width="40" height="2" fill="#fff" opacity="0.5"/></pattern>')
    out.append('<pattern id="stripes" width="40" height="28" patternUnits="userSpaceOnUse">'
               '<rect y="0" width="40" height="12" fill="#fff" opacity="0.75"/></pattern>')
    out.append('</defs>')
    return "".join(out)


def background(p: Pen) -> str:
    t = p.t
    bg = [f'<rect width="{W}" height="{W}" fill="{t["bgc"]}"/>']
    kind = t["bg"]
    if kind == "dots":
        bg.append(f'<circle cx="{CX}" cy="360" r="300" fill="#fbe0d2"/>')
    elif kind == "wash":
        bg.append('<g filter="url(#blur30)" opacity="0.75"><ellipse cx="200" cy="200" rx="260" ry="180" fill="#f4c99a"/>'
                  '<ellipse cx="600" cy="260" rx="240" ry="200" fill="#a9cbe0"/>'
                  '<ellipse cx="380" cy="700" rx="420" ry="160" fill="#c9dcb0"/></g>')
    elif kind == "paper":
        bg.append(f'<rect width="{W}" height="{W}" filter="url(#grain)" opacity="0.5"/>')
    elif kind == "halftone":
        bg.append(f'<rect width="{W}" height="{W}" fill="url(#dots)" opacity="0.55"/>')
        bg.append(f'<rect x="12" y="12" width="{W-24}" height="{W-24}" fill="none" stroke="#111" stroke-width="6"/>')
    elif kind == "hanji":
        bg.append(f'<rect width="{W}" height="{W}" filter="url(#grain)"/>')
        bg.append('<path d="M0 640 Q120 600 230 630 T470 620 T768 610 L768 768 L0 768Z" fill="#d9c9a3" opacity="0.6"/>')
        bg.append('<path d="M560 180 Q600 120 650 170 Q690 140 720 190" fill="none" stroke="#9b8b70" stroke-width="3" opacity="0.6"/>')
    elif kind == "mountains":
        bg.append('<path d="M0 420 L120 250 L210 360 L330 180 L470 380 L560 260 L768 430 L768 768 L0 768Z" fill="#b9c3c4"/>')
        bg.append('<path d="M0 520 L150 380 L260 470 L420 330 L560 480 L680 400 L768 470 L768 768 L0 768Z" fill="#97a3a5"/>')
        bg.append(f'<rect y="430" width="{W}" height="200" fill="#eef1f0" opacity="0.55" filter="url(#blur30)"/>')
    return "".join(bg)


# ---------- 사람 ----------

def hair_back(p: Pen, style: str, col: str) -> str:
    s = p.st()
    h = p.c(col)
    if style in ("long", "wavy", "straight", "wuxia_long"):
        edge = "Q 250 560 236 600" if style != "wavy" else "Q 220 470 250 520 Q 220 560 245 610"
        edge2 = "Q 518 560 532 600" if style != "wavy" else "Q 548 470 518 520 Q 548 560 523 610"
        return (f'<path d="M {CX} 150 Q 240 150 248 300 {edge} L 300 600 L 300 430 L 468 430 L 468 600 L 532 600 '
                f'{edge2.replace("Q 518 560 532 600", "Q 518 560 520 600") if False else ""} Q 528 150 {CX} 150 Z" fill="{h}" {s}/>'
                if False else
                f'<path d="M {CX} 152 C 236 152 246 300 244 380 C 242 470 236 540 232 610 L 536 610 C 532 540 526 470 524 380 '
                f'C 522 300 532 152 {CX} 152 Z" fill="{h}" {s}/>'
                + (f'<path d="M 232 610 Q 220 580 240 560 M 536 610 Q 548 580 528 560" fill="none" {s}/>' if style == "wavy" else ""))
    if style in ("bob",):
        return f'<path d="M {CX} 150 C 232 150 236 300 240 420 Q 300 446 {CX} 440 Q 468 446 528 420 C 532 300 536 150 {CX} 150 Z" fill="{h}" {s}/>'
    if style == "ponytail":
        return f'<path d="M 470 200 C 600 210 610 380 560 470 C 540 420 540 330 480 260 Z" fill="{h}" {s}/>'
    if style == "pigtails":
        return (f'<ellipse cx="236" cy="330" rx="46" ry="70" fill="{h}" {s}/>'
                f'<ellipse cx="532" cy="330" rx="46" ry="70" fill="{h}" {s}/>')
    if style == "bun":
        return f'<circle cx="{CX}" cy="150" r="52" fill="{h}" {s}/>'
    if style == "chignon":
        return (f'<ellipse cx="{CX + 118}" cy="380" rx="44" ry="34" fill="{h}" {s}/>'
                f'<line x1="{CX + 70}" y1="372" x2="{CX + 190}" y2="386" stroke="{p.c("#c9a227")}" stroke-width="8" stroke-linecap="round"/>')
    if style == "braid":
        return ""
    if style == "wuxia_high":
        return (f'<path d="M {CX} 140 C 470 140 520 230 540 330 C 560 430 540 520 520 600 L 470 600 C 500 500 500 400 470 300 Z" fill="{h}" {s}/>')
    return ""


def hair_front(p: Pen, style: str, col: str) -> str:
    s = p.st()
    h = p.c(col)
    if style == "none":
        return ""
    cap = (f'<path d="M 262 300 C 250 170 330 150 {CX} 150 C 438 150 518 170 506 300 '
           f'C 470 230 420 215 {CX} 215 C 348 215 298 230 262 300 Z" fill="{h}" {s}/>')
    if style in ("long", "straight", "bob", "wavy"):
        return (f'<path d="M 258 320 C 246 170 330 148 {CX} 148 C 438 148 522 170 510 320 '
                f'C 490 250 450 225 420 215 Q 400 250 330 255 Q 290 262 258 320 Z" fill="{h}" {s}/>')
    if style == "short":
        return cap
    if style == "messy":
        return (f'<path d="M 260 300 L 250 230 L 285 245 L 290 180 L 330 210 L 350 150 L 385 200 L 420 150 L 440 205 '
                f'L 485 175 L 485 240 L 518 232 L 508 300 C 470 235 420 220 {CX} 222 C 348 220 298 235 260 300 Z" fill="{h}" {s}/>')
    if style == "curly":
        circles = "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{h}" {s}/>' for x, y, r in
                          [(270, 270, 34), (285, 210, 38), (335, 172, 40), (395, 160, 42), (450, 178, 40), (495, 220, 36), (505, 275, 32)])
        return circles + f'<path d="M 268 270 Q {CX} 200 500 270 L 500 230 Q {CX} 170 268 230 Z" fill="{h}"/>'
    if style in ("buzz", "crew"):
        thick = 205 if style == "crew" else 195
        return (f'<path d="M 266 290 C 256 175 330 160 {CX} 160 C 438 160 512 175 502 290 '
                f'C 470 {thick + 10} 420 {thick} {CX} {thick} C 348 {thick} 298 {thick + 10} 266 290 Z" fill="{h}" {s}/>')
    if style == "ponytail":
        return cap + f'<circle cx="482" cy="228" r="13" fill="{p.c("red")}" {p.st(3)}/>'
    if style == "pigtails":
        return (cap + f'<line x1="{CX}" y1="152" x2="{CX}" y2="214" {p.st(4)}/>'
                f'<circle cx="262" cy="282" r="12" fill="{p.c("yellow")}" {p.st(3)}/><circle cx="506" cy="282" r="12" fill="{p.c("yellow")}" {p.st(3)}/>')
    if style == "bun":
        return cap
    if style in ("chignon", "braid"):  # 가운데 가르마, 매끈하게 빗은 머리
        part = (f'<path d="M 258 315 C 248 170 330 150 {CX} 150 C 438 150 520 170 510 315 '
                f'C 490 240 440 205 {CX} 200 C 328 205 278 240 258 315 Z" fill="{h}" {s}/>'
                f'<path d="M {CX} 152 L {CX} 200" {p.st(3)}/>')
        if style == "braid":
            beads = "".join(f'<ellipse cx="{500 + i * 4}" cy="{400 + i * 46}" rx="24" ry="28" fill="{h}" {s}/>' for i in range(5))
            ribbon = f'<path d="M 500 630 L 470 700 L 520 690 L 540 700 L 530 630 Z" fill="{p.c("crimson")}" {s}/>'
            return part + beads + ribbon
        return part
    if style == "topknot":
        return (f'<path d="M 262 300 C 250 170 330 150 {CX} 150 C 438 150 518 170 506 300 C 470 225 420 210 {CX} 210 '
                f'C 348 210 298 225 262 300 Z" fill="{h}" {s}/>'
                f'<ellipse cx="{CX}" cy="128" rx="26" ry="30" fill="{h}" {s}/>'
                f'<path d="M 262 262 Q {CX} 200 506 262" fill="none" stroke="{p.c("black")}" stroke-width="16"/>')
    if style == "wuxia_high":
        return (cap + f'<path d="M 340 150 Q {CX} 70 428 150 Z" fill="{h}" {s}/>'
                f'<rect x="352" y="120" width="64" height="18" rx="6" fill="{p.c("#c9a227")}" {p.st(3)}/>')
    if style == "wuxia_long":
        return (f'<path d="M 258 320 C 246 170 330 148 {CX} 148 C 438 148 522 170 510 320 C 480 240 430 215 {CX} 210 '
                f'C 338 215 288 240 258 320 Z" fill="{h}" {s}/>'
                f'<path d="M 330 150 Q {CX} 90 438 150 Z" fill="{h}" {s}/>'
                f'<line x1="300" y1="110" x2="468" y2="140" stroke="{p.c("#c9a227")}" stroke-width="7" stroke-linecap="round"/>')
    return cap


def face(p: Pen, eyes: str, skin: str, mouth: str = "smile", brow: str = "black", blush: bool = True,
         extra: str = "") -> str:
    s = p.st()
    out = [f'<ellipse cx="{CX - RX + 2}" cy="{CY + 20}" rx="18" ry="26" fill="{skin}" {s}/>',
           f'<ellipse cx="{CX + RX - 2}" cy="{CY + 20}" rx="18" ry="26" fill="{skin}" {s}/>',
           f'<ellipse cx="{CX}" cy="{CY}" rx="{RX}" ry="{RY}" fill="{skin}" {s}/>']
    ex, ey = 44, CY + 20
    bc = p.c(brow)
    for side in (-1, 1):
        x = CX + side * ex
        if eyes == "round":
            out.append(f'<ellipse cx="{x}" cy="{ey}" rx="13" ry="17" fill="#2a2320"/><circle cx="{x + 4}" cy="{ey - 6}" r="5" fill="#fff"/>')
        elif eyes == "lash":
            out.append(f'<ellipse cx="{x}" cy="{ey}" rx="13" ry="17" fill="#2a2320"/><circle cx="{x + 4}" cy="{ey - 6}" r="5" fill="#fff"/>'
                       f'<path d="M {x - 15} {ey - 14} Q {x} {ey - 26} {x + 15} {ey - 14} L {x + side * 22} {ey - 22}" fill="none" {p.st(4)}/>')
        elif eyes == "smile":
            out.append(f'<path d="M {x - 15} {ey + 4} Q {x} {ey - 14} {x + 15} {ey + 4}" fill="none" {p.st(6)}/>')
        elif eyes == "sharp":
            out.append(f'<path d="M {x - 18} {ey + 2} Q {x} {ey - 12} {x + 18} {ey - 2 * side} Q {x} {ey + 8} {x - 18} {ey + 2} Z" fill="#2a2320"/>'
                       f'<circle cx="{x + 2}" cy="{ey - 2}" r="3" fill="#fff"/>')
        elif eyes == "dot":
            out.append(f'<circle cx="{x}" cy="{ey}" r="9" fill="#2a2320"/>')
        by = ey - (34 if eyes != "sharp" else 26)
        tilt = -8 if eyes == "sharp" else 0
        out.append(f'<path d="M {x - 18} {by + (tilt if side < 0 else 0) + 2} Q {x} {by - 8} {x + 18} {by + (tilt if side > 0 else 0) + 2}" '
                   f'fill="none" stroke="{bc}" stroke-width="7" stroke-linecap="round"/>')
    out.append(f'<path d="M {CX - 4} {CY + 50} Q {CX} {CY + 58} {CX + 5} {CY + 52}" fill="none" {p.st(3)}/>')
    my = CY + 82
    if mouth == "smile":
        out.append(f'<path d="M {CX - 26} {my} Q {CX} {my + 24} {CX + 26} {my}" fill="none" {p.st(5)}/>')
    elif mouth == "open":
        out.append(f'<path d="M {CX - 24} {my - 2} Q {CX} {my + 34} {CX + 24} {my - 2} Z" fill="#b5463f" {p.st(4)}/>')
    elif mouth == "flat":
        out.append(f'<path d="M {CX - 18} {my + 4} Q {CX} {my + 10} {CX + 18} {my + 4}" fill="none" {p.st(5)}/>')
    elif mouth == "smirk":
        out.append(f'<path d="M {CX - 20} {my + 6} Q {CX + 6} {my + 14} {CX + 24} {my - 2}" fill="none" {p.st(5)}/>')
    if blush and not p.t.get("gray"):
        out.append(f'<ellipse cx="{CX - 72}" cy="{CY + 62}" rx="20" ry="11" fill="#f08a8a" opacity="0.45"/>'
                   f'<ellipse cx="{CX + 72}" cy="{CY + 62}" rx="20" ry="11" fill="#f08a8a" opacity="0.45"/>')
    out.append(extra)
    return "".join(out)


TORSO = "M 186 768 L 196 580 Q 206 492 300 474 L 468 474 Q 562 492 572 580 L 582 768 Z"
ROBE = "M 140 768 L 160 580 Q 176 490 296 470 L 472 470 Q 592 490 608 580 L 628 768 Z"


def neck(p: Pen, skin: str) -> str:
    return f'<path d="M 346 410 L 346 490 L 422 490 L 422 410 Z" fill="{skin}" {p.st()}/>'


def clothes(p: Pen, kind: str, col: str, skin: str) -> str:
    s = p.st()
    c = p.c(col)
    if kind == "hoodie":
        return (f'<path d="M 270 500 Q {CX} 560 498 500 Q 470 455 {CX} 455 Q 298 455 270 500 Z" fill="{c}" {s}/>'
                f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 300 474 Q {CX} 540 468 474" fill="none" {s}/>'
                f'<line x1="356" y1="515" x2="350" y2="600" {p.st(4)}/><line x1="412" y1="515" x2="418" y2="600" {p.st(4)}/>'
                f'<path d="M 290 680 L 478 680 L 470 768 L 298 768 Z" fill="none" {p.st(4)}/>')
    if kind == "cardigan":
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 320 474 L {CX} 600 L 448 474 Z" fill="{p.c("white")}" {s}/>'
                f'<line x1="{CX}" y1="600" x2="{CX}" y2="768" {s}/>'
                + "".join(f'<circle cx="{CX + 18}" cy="{y}" r="7" fill="{p.c("cream")}" {p.st(3)}/>' for y in (630, 680, 730)))
    if kind == "sweater":
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 330 476 Q {CX} 520 438 476" fill="none" {p.st(10)}/>'
                f'<path d="M 330 476 Q {CX} 520 438 476" fill="none" stroke="{c}" stroke-width="4"/>')
    if kind == "flannel":
        return (f'<path d="{TORSO}" fill="{c}" {s}/><path d="{TORSO}" fill="url(#plaid)"/>'
                f'<path d="M 320 470 L {CX} 540 L 340 560 L 300 490 Z M 448 470 L {CX} 540 L 428 560 L 468 490 Z" fill="{c}" {s}/>'
                f'<line x1="{CX}" y1="540" x2="{CX}" y2="768" {s}/>')
    if kind in ("blouse", "shirt"):
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 330 470 L {CX} 520 L 360 560 L 310 490 Z M 438 470 L {CX} 520 L 408 560 L 458 490 Z" fill="{c}" {s}/>'
                f'<line x1="{CX}" y1="520" x2="{CX}" y2="768" {p.st(3)}/>'
                + "".join(f'<circle cx="{CX + 10}" cy="{y}" r="5" fill="{p.c("cream")}" {p.st(2)}/>' for y in (580, 640, 700)))
    if kind == "dress":
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 320 476 Q {CX} 540 448 476" fill="{skin}" {s}/>'
                f'<path d="M 220 700 Q {CX} 720 548 700" fill="none" {p.st(4)}/>')
    if kind == "jacket":
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 330 476 L {CX} 560 L 438 476 Z" fill="{p.c("white")}" {s}/>'
                f'<path d="M 320 474 L 360 600 L 330 620 L 296 490 Z M 448 474 L 408 600 L 438 620 L 472 490 Z" fill="{c}" {s}/>'
                f'<line x1="{CX}" y1="560" x2="{CX}" y2="768" {s}/>'
                f'<rect x="230" y="660" width="70" height="44" rx="6" fill="none" {p.st(4)}/>'
                f'<rect x="468" y="660" width="70" height="44" rx="6" fill="none" {p.st(4)}/>')
    if kind == "raincoat":
        return (f'<path d="M 260 500 Q {CX} 440 508 500 Q 470 445 {CX} 445 Q 298 445 260 500 Z" fill="{c}" {s}/>'
                f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 320 474 L {CX} 530 L 448 474" fill="none" {s}/><line x1="{CX}" y1="530" x2="{CX}" y2="768" {s}/>'
                + "".join(f'<circle cx="{CX + 24}" cy="{y}" r="9" fill="{p.c("white")}" {p.st(3)}/>' for y in (580, 640, 700)))
    if kind == "overalls":
        return (f'<path d="{TORSO}" fill="{p.c("white")}" {s}/>'
                f'<path d="M 300 600 L 468 600 L 490 768 L 278 768 Z" fill="{c}" {s}/>'
                f'<path d="M 300 600 L 290 478 L 320 476 L 330 600 Z M 468 600 L 478 478 L 448 476 L 438 600 Z" fill="{c}" {s}/>'
                f'<circle cx="318" cy="612" r="9" fill="{p.c("yellow")}" {p.st(3)}/><circle cx="450" cy="612" r="9" fill="{p.c("yellow")}" {p.st(3)}/>'
                f'<rect x="344" y="640" width="80" height="56" rx="6" fill="none" {p.st(4)}/>')
    if kind == "striped":
        return (f'<path d="{TORSO}" fill="{c}" {s}/><path d="{TORSO}" fill="url(#stripes)"/>'
                f'<path d="M 330 476 Q {CX} 520 438 476" fill="none" {p.st(8)}/>'
                f'<path d="{TORSO}" fill="none" {s}/>')
    if kind == "hanbok_w":  # 짧은 저고리 + 가슴까지 올라온 치마
        jeo = p.c("cream")
        return (f'<path d="M 170 768 L 190 600 Q 220 560 300 560 L 468 560 Q 548 560 578 600 L 598 768 Z" fill="{c}" {s}/>'
                f'<path d="M 196 600 Q 206 492 300 474 L 468 474 Q 562 492 572 600 Q 540 590 520 600 L 250 600 Q 226 590 196 600 Z" fill="{jeo}" {s}/>'
                f'<path d="M 330 474 L {CX} 560 L 438 474" fill="none" {s}/>'
                f'<path d="M 318 474 L {CX} 566 L 404 520" fill="none" stroke="{p.c("white")}" stroke-width="14"/>'
                f'<path d="M 318 474 L {CX} 566 L 404 520" fill="none" {p.st(2)}/>'
                f'<path d="M 400 575 Q 430 560 440 580 Q 430 600 400 590 Z M 400 585 L 380 690 L 405 690 L 412 595 Z M 405 585 L 440 680 L 460 670 L 418 588 Z" fill="{p.c("crimson")}" {p.st(3)}/>'
                + "".join(f'<path d="M {x} 610 Q {x + 6} 690 {x - 4} 768" fill="none" {p.st(2)} opacity="0.7"/>' for x in (240, 300, 470, 530)))
    if kind == "hanbok_m":  # 저고리 + 동정 + 고름
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 320 474 L {CX} 600 L 470 520" fill="none" stroke="{p.c("white")}" stroke-width="18"/>'
                f'<path d="M 320 474 L {CX} 600 L 470 520" fill="none" {p.st(2)}/>'
                f'<path d="M 440 474 L {CX} 540" fill="none" {s}/>'
                f'<path d="M 400 610 Q 430 596 440 616 Q 430 636 400 626 Z M 402 620 L 386 740 L 410 740 L 414 630 Z" fill="{p.c(col)}" {p.st(3)}/>'
                f'<path d="M 196 700 L 572 700" fill="none" {p.st(3)} opacity="0.5"/>')
    if kind == "robe":  # 무협 도포: 넓은 교차 깃과 허리띠
        trim = p.c("white") if col not in ("white",) else p.c("#9aa7b5")
        return (f'<path d="{ROBE}" fill="{c}" {s}/>'
                f'<path d="M 310 470 L {CX} 640 L 480 520" fill="none" stroke="{trim}" stroke-width="22"/>'
                f'<path d="M 458 470 L {CX} 560" fill="none" stroke="{trim}" stroke-width="22"/>'
                f'<path d="M 310 470 L {CX} 640 L 480 520 M 458 470 L {CX} 560" fill="none" {p.st(3)}/>'
                f'<path d="M 150 660 L 618 660 L 622 710 L 146 710 Z" fill="{p.c("black") if col != "black" else p.c("crimson")}" {s}/>'
                f'<path d="M 170 600 Q 220 650 200 768 M 598 600 Q 548 650 568 768" fill="none" {p.st(3)} opacity="0.6"/>')
    if kind in EXTRA_CLOTHES:
        return EXTRA_CLOTHES[kind](p, c, skin)
    return f'<path d="{TORSO}" fill="{c}" {s}/>'


EXTRA_CLOTHES: dict = {}  # jobs.py가 채운다
EXTRA_ACC: dict = {}


def accessory(p: Pen, kind: str, skin: str) -> str:
    s = p.st()
    if kind == "glasses":
        return (f'<circle cx="{CX - 44}" cy="{CY + 20}" r="30" fill="none" {p.st(5)}/>'
                f'<circle cx="{CX + 44}" cy="{CY + 20}" r="30" fill="none" {p.st(5)}/>'
                f'<line x1="{CX - 14}" y1="{CY + 18}" x2="{CX + 14}" y2="{CY + 18}" {p.st(5)}/>')
    if kind == "beret":
        return (f'<path d="M 250 210 Q 260 120 {CX} 118 Q 520 120 520 200 Q 470 230 {CX} 222 Q 300 230 250 210 Z" fill="{p.c("#333")}" {s}/>'
                f'<line x1="{CX}" y1="118" x2="{CX + 6}" y2="98" {p.st(6)}/>')
    if kind == "pearls":
        return "".join(f'<circle cx="{CX + 70 * __import__("math").cos(a / 10)}" cy="{486 + 34 * __import__("math").sin(a / 10)}" r="9" fill="#fafafa" {p.st(2)}/>'
                       for a in range(2, 30, 3))
    if kind == "bowtie":
        return (f'<path d="M {CX} 496 L 334 470 L 334 522 Z M {CX} 496 L 434 470 L 434 522 Z" fill="{p.c("red")}" {s}/>'
                f'<circle cx="{CX}" cy="496" r="11" fill="{p.c("red")}" {s}/>')
    if kind == "gat":
        brim = p.c("black")
        return (f'<ellipse cx="{CX}" cy="200" rx="230" ry="44" fill="{brim}" fill-opacity="0.78" {s}/>'
                f'<path d="M 316 196 L 326 86 Q {CX} 70 442 86 L 452 196 Z" fill="{brim}" fill-opacity="0.85" {s}/>'
                f'<path d="M 270 230 Q 280 400 330 470 M 498 230 Q 488 400 438 470" fill="none" stroke="{brim}" stroke-width="4"/>')
    if kind == "long sword":
        return (f'<line x1="560" y1="768" x2="640" y2="250" stroke="#cfd6db" stroke-width="16" stroke-linecap="round"/>'
                f'<line x1="560" y1="768" x2="640" y2="250" {p.st(3)}/>'
                f'<line x1="585" y1="610" x2="660" y2="625" {p.st(14)}/><rect x="583" y="560" width="22" height="54" fill="{p.c("crimson")}" {p.st(3)} transform="rotate(9 594 587)"/>')
    if kind == "broadsword":
        return (f'<path d="M 150 300 L 210 290 L 250 640 L 190 650 Z" fill="#cfd6db" {s}/>'
                f'<line x1="160" y1="650" x2="290" y2="630" {p.st(14)}/><rect x="198" y="650" width="26" height="90" fill="{p.c("dark brown")}" {s} transform="rotate(-9 211 695)"/>')
    if kind == "folding fan":
        ribs = "".join(f'<line x1="260" y1="700" x2="{260 + 150 * __import__("math").cos(a)}" y2="{700 - 150 * __import__("math").sin(a)}" {p.st(2)}/>'
                       for a in [0.5 + i * 0.25 for i in range(8)])
        return (f'<path d="M 260 700 L {260 + 150 * 0.8776} {700 - 150 * 0.4794} A 150 150 0 0 0 {260 + 150 * -0.4161} {700 - 150 * 0.9093} Z" fill="{p.c("white")}" {s}/>'
                + ribs + f'<circle cx="250" cy="712" r="26" fill="{skin}" {s}/>')
    if kind == "bamboo flute":
        return (f'<line x1="230" y1="720" x2="470" y2="560" stroke="{p.c("#b8a35a")}" stroke-width="18" stroke-linecap="round"/>'
                f'<line x1="230" y1="720" x2="470" y2="560" {p.st(3)}/>'
                + "".join(f'<circle cx="{300 + i * 36}" cy="{673 - i * 24}" r="4" fill="#333"/>' for i in range(4))
                + f'<circle cx="260" cy="705" r="26" fill="{skin}" {s}/><circle cx="430" cy="590" r="26" fill="{skin}" {s}/>')
    if kind == "mustache":
        return f'<path d="M {CX - 40} {CY + 72} Q {CX - 20} {CY + 60} {CX} {CY + 70} Q {CX + 20} {CY + 60} {CX + 40} {CY + 72}" fill="none" stroke="{p.c("black")}" stroke-width="6" stroke-linecap="round"/>'
    if kind == "scar":
        return f'<path d="M {CX + 60} {CY - 30} L {CX + 80} {CY + 10}" {p.st(4)}/>'
    if kind in EXTRA_ACC:
        return EXTRA_ACC[kind](p, skin)
    return ""


def person(theme: str, d: dict) -> str:
    p = Pen(theme)
    skin = p.c(SKIN[d.get("skin", "light")])
    parts = [background(p)]
    body = [hair_back(p, d["hair"], d["hair_color"]), neck(p, skin), clothes(p, d["clothes"], d["color"], skin)]
    for a in d.get("under", []):
        body.append(accessory(p, a, skin))
    body.append(face(p, d.get("eyes", "round"), skin, d.get("mouth", "smile"), d["hair_color"]))
    body.append(hair_front(p, d["hair"], d["hair_color"]))
    for a in d.get("acc", []):
        body.append(accessory(p, a, skin))
    filt = p.t.get("filt")
    g = f'<g filter="url(#{filt})">' if filt else "<g>"
    parts.append(g + "".join(body) + "</g>")
    if theme == "NEWSPAPER":
        parts.append(f'<path d="{TORSO}" fill="url(#hatch)" opacity="0.18"/>')
    return svg(p, "".join(parts))


# ---------- 동물 ----------

ANIMAL = {
    "rabbit": dict(fur="#f6efe9", inner="#f4b6c2"),
    "bear": dict(fur="#a7754c", inner="#e8c9a4"),
    "fox": dict(fur="#ef8f3e", inner="#fff6ea"),
    "panda": dict(fur="#fafafa", inner="#2a2a2a"),
    "cat": dict(fur="#c9c4be", inner="#f4b6c2"),
    "deer": dict(fur="#c98f5a", inner="#f6e1c6"),
    "dog": dict(fur="#e7c48f", inner="#9c6a3c"),
    "squirrel": dict(fur="#b9774a", inner="#f1d9bd"),
}


def animal(d: dict) -> str:
    p = Pen("ANIMALS")
    s = p.st()
    a = d["animal"]
    fur, inner = ANIMAL[a]["fur"], ANIMAL[a]["inner"]
    o = [background(p)]
    # 뒤에 오는 것: 귀, 꼬리
    if a == "squirrel":
        o.append(f'<path d="M 520 768 C 700 700 720 480 640 400 C 590 350 520 380 560 440 C 610 520 560 640 470 700 Z" fill="{fur}" {s}/>'
                 f'<path d="M 600 470 C 640 540 620 640 540 700" fill="none" stroke="{inner}" stroke-width="18" stroke-linecap="round"/>')
    if a == "rabbit":
        for x, r in ((330, -10), (438, 10)):
            o.append(f'<g transform="rotate({r} {x} 180)"><ellipse cx="{x}" cy="110" rx="38" ry="110" fill="{fur}" {s}/>'
                     f'<ellipse cx="{x}" cy="115" rx="18" ry="80" fill="{inner}"/></g>')
    elif a in ("bear", "panda"):
        ec = inner if a == "panda" else fur
        for x in (276, 492):
            o.append(f'<circle cx="{x}" cy="190" r="48" fill="{ec}" {s}/>')
            if a == "bear":
                o.append(f'<circle cx="{x}" cy="190" r="24" fill="{inner}"/>')
    elif a in ("fox", "cat", "squirrel"):
        for side in (-1, 1):
            x = CX + side * 90
            o.append(f'<path d="M {x - 50} 230 L {x + side * 20} {90 if a == "fox" else 120} L {x + 50} 230 Z" fill="{fur}" {s}/>'
                     f'<path d="M {x - 26} 220 L {x + side * 14} {130 if a == "fox" else 150} L {x + 26} 220 Z" fill="{inner if a != "fox" else "#3a2a22"}"/>')
            if a == "squirrel":
                o.append(f'<path d="M {x + side * 14} 120 l {side * 8} -30 l {side * -16} 14 Z" fill="{fur}" {s}/>')
    elif a == "deer":
        for side in (-1, 1):
            o.append(f'<path d="M {CX + side * 60} 190 L {CX + side * 80} 90 M {CX + side * 75} 120 L {CX + side * 120} 95 M {CX + side * 72} 140 L {CX + side * 40} 110" '
                     f'fill="none" stroke="#7a5532" stroke-width="12" stroke-linecap="round"/>')
            o.append(f'<ellipse cx="{CX + side * 140}" cy="230" rx="52" ry="26" fill="{fur}" {s} transform="rotate({side * 20} {CX + side * 140} 230)"/>')
    elif a == "dog":
        pass
    # 몸과 옷
    o.append(f'<path d="M 346 410 L 346 490 L 422 490 L 422 410 Z" fill="{fur}" {s}/>')
    o.append(clothes(p, d["clothes"], d["color"], fur))
    # 머리
    o.append(f'<ellipse cx="{CX}" cy="{CY}" rx="{RX + 8}" ry="{RY - 6}" fill="{fur}" {s}/>')
    if a == "fox":
        o.append(f'<path d="M {CX - 120} {CY + 10} Q {CX - 60} {CY + 140} {CX} {CY + 120} Q {CX + 60} {CY + 140} {CX + 120} {CY + 10} Q {CX + 40} {CY + 60} {CX} {CY + 50} Q {CX - 40} {CY + 60} {CX - 120} {CY + 10} Z" fill="{inner}"/>')
    if a == "panda":
        for side in (-1, 1):
            o.append(f'<ellipse cx="{CX + side * 46}" cy="{CY + 14}" rx="34" ry="42" fill="{inner}" transform="rotate({side * -25} {CX + side * 46} {CY + 14})"/>')
    if a in ("bear", "dog", "deer", "squirrel"):
        o.append(f'<ellipse cx="{CX}" cy="{CY + 66}" rx="62" ry="46" fill="{ANIMAL["bear"]["inner"] if a != "dog" else "#f7e6c8"}" {s}/>')
    if a == "dog":
        for side in (-1, 1):
            o.append(f'<path d="M {CX + side * 100} 200 C {CX + side * 190} 210 {CX + side * 180} 360 {CX + side * 130} 380 C {CX + side * 110} 320 {CX + side * 100} 260 {CX + side * 100} 200 Z" fill="{inner}" {s}/>')
    if a == "deer":
        for x, y in ((CX - 80, CY - 60), (CX + 70, CY - 70), (CX - 40, CY - 95), (CX + 30, CY - 100)):
            o.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#fff4e4"/>')
    # 눈·코·입
    for side in (-1, 1):
        x = CX + side * 46
        o.append(f'<ellipse cx="{x}" cy="{CY + 14}" rx="13" ry="16" fill="#241c19"/><circle cx="{x + 4}" cy="{CY + 8}" r="5" fill="#fff"/>')
    o.append(f'<ellipse cx="{CX}" cy="{CY + 52}" rx="16" ry="11" fill="#3a2a26"/>')
    o.append(f'<path d="M {CX} {CY + 62} L {CX} {CY + 74} M {CX - 22} {CY + 80} Q {CX - 10} {CY + 92} {CX} {CY + 74} Q {CX + 10} {CY + 92} {CX + 22} {CY + 80}" fill="none" {p.st(4)}/>')
    o.append(f'<ellipse cx="{CX - 80}" cy="{CY + 56}" rx="20" ry="11" fill="#f39a9a" opacity="0.55"/><ellipse cx="{CX + 80}" cy="{CY + 56}" rx="20" ry="11" fill="#f39a9a" opacity="0.55"/>')
    if a == "cat":
        for side in (-1, 1):
            for dy in (-6, 8):
                o.append(f'<line x1="{CX + side * 40}" y1="{CY + 62 + dy}" x2="{CX + side * 120}" y2="{CY + 54 + dy * 2}" {p.st(3)}/>')
        o.append(f'<path d="M {CX - 20} {CY - 100} l 10 30 M {CX} {CY - 104} l 0 32 M {CX + 20} {CY - 100} l -10 30" {p.st(5)}/>')
    # 소품
    acc = d.get("acc")
    if acc == "scarf":
        o.append(f'<path d="M 300 460 Q {CX} 510 468 460 L 476 500 Q {CX} 550 292 500 Z" fill="{p.c(d["acc_color"])}" {s}/>'
                 f'<path d="M 420 500 L 440 620 L 480 610 L 452 492 Z" fill="{p.c(d["acc_color"])}" {s}/>')
    elif acc == "bowtie":
        o.append(accessory(p, "bowtie", fur).replace(p.c("red"), p.c(d["acc_color"])))
    elif acc == "bandana":
        o.append(f'<path d="M 296 470 L 472 470 L {CX} 580 Z" fill="{p.c(d["acc_color"])}" {s}/>'
                 + "".join(f'<circle cx="{x}" cy="{y}" r="5" fill="#fff"/>' for x, y in ((350, 490), (418, 490), (384, 530))))
    elif acc == "backpack":
        o.append(f'<path d="M 270 520 L 300 768 M 498 520 L 468 768" fill="none" stroke="{p.c(d["acc_color"])}" stroke-width="30"/>'
                 f'<path d="M 270 520 L 300 768 M 498 520 L 468 768" fill="none" {p.st(3)}/>')
    for extra in d.get("extra", []):
        o.append(accessory(p, extra, fur))
    return svg(p, "".join(o))


def svg(p: Pen, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}">'
            + defs(p) + body + "</svg>")


# ---------- 캐릭터 목록 (catalog.yaml의 look과 같다) ----------

CAST = {
    "DAILY": {
        "지우": dict(hair="long", hair_color="black", clothes="hoodie", color="yellow", eyes="lash", skin="light"),
        "하린": dict(hair="bob", hair_color="brown", clothes="cardigan", color="green", eyes="round", mouth="open", skin="warm"),
        "도윤": dict(hair="short", hair_color="black", clothes="sweater", color="navy", eyes="dot", skin="warm"),
        "준호": dict(hair="curly", hair_color="brown", clothes="flannel", color="red", eyes="smile", mouth="open", skin="tan"),
    },
    "WATERCOLOR": {
        "소라": dict(hair="wavy", hair_color="auburn", clothes="blouse", color="white", eyes="lash", skin="light"),
        "유나": dict(hair="ponytail", hair_color="black", clothes="dress", color="sky blue", eyes="round", skin="warm"),
        "태오": dict(hair="short", hair_color="brown", clothes="shirt", color="white", eyes="round", mouth="flat", skin="warm"),
        "시우": dict(hair="messy", hair_color="black", clothes="jacket", color="olive", eyes="dot", mouth="smirk", skin="light"),
    },
    "PICTURE_BOOK": {
        "콩이": dict(hair="pigtails", hair_color="black", clothes="dress", color="red", eyes="round", mouth="open", skin="light"),
        "봄이": dict(hair="bob", hair_color="brown", clothes="raincoat", color="yellow", eyes="smile", skin="warm"),
        "단우": dict(hair="buzz", hair_color="black", clothes="overalls", color="blue", eyes="dot", mouth="open", skin="warm"),
        "해솔": dict(hair="curly", hair_color="dark brown", clothes="striped", color="green", eyes="round", skin="tan"),
    },
    "NEWSPAPER": {
        "영자": dict(hair="bun", hair_color="black", clothes="blouse", color="gray", eyes="lash", acc=["pearls"]),
        "미숙": dict(hair="straight", hair_color="black", clothes="dress", color="white", eyes="round", acc=["beret"]),
        "철수": dict(hair="curly", hair_color="black", clothes="shirt", color="white", eyes="dot", mouth="open", acc=["glasses"]),
        "만식": dict(hair="crew", hair_color="black", clothes="jacket", color="gray", eyes="smile", acc=["bowtie"]),
    },
    "JOSEON": {
        "연이": dict(hair="braid", hair_color="black", clothes="hanbok_w", color="pink", eyes="lash", skin="light"),
        "단아": dict(hair="chignon", hair_color="black", clothes="hanbok_w", color="purple", eyes="round", mouth="flat", skin="light"),
        "김선비": dict(hair="topknot", hair_color="black", clothes="hanbok_m", color="blue", eyes="sharp", mouth="flat", skin="warm", acc=["mustache", "gat"]),
        "이도령": dict(hair="topknot", hair_color="black", clothes="hanbok_m", color="white", eyes="round", skin="light"),
    },
    "WUXIA": {
        "설화": dict(hair="wuxia_long", hair_color="black", clothes="robe", color="white", eyes="sharp", mouth="flat", skin="light", acc=["long sword"]),
        "홍련": dict(hair="wuxia_long", hair_color="dark brown", clothes="robe", color="red", eyes="lash", mouth="smirk", skin="light", acc=["folding fan"]),
        "무진": dict(hair="wuxia_high", hair_color="black", clothes="robe", color="black", eyes="sharp", mouth="flat", skin="tan", acc=["scar", "broadsword"]),
        "청운": dict(hair="wuxia_high", hair_color="black", clothes="robe", color="blue", eyes="smile", skin="warm", acc=["bamboo flute"]),
    },
}

ANIMALS = {
    "토토": dict(animal="rabbit", clothes="sweater", color="cream", acc="scarf", acc_color="pink"),
    "곰곰": dict(animal="bear", clothes="sweater", color="green"),
    "여울": dict(animal="fox", clothes="raincoat", color="yellow"),
    "판다랑": dict(animal="panda", clothes="shirt", color="white", acc="bowtie", acc_color="red"),
    "나비": dict(animal="cat", clothes="overalls", color="blue"),
    "사슴이": dict(animal="deer", clothes="dress", color="purple"),
    "멍구": dict(animal="dog", clothes="sweater", color="cream", acc="bandana", acc_color="orange"),
    "도토리": dict(animal="squirrel", clothes="sweater", color="white", acc="backpack", acc_color="teal"),
}


def main(out: Path) -> None:
    for theme, cast in CAST.items():
        for name, d in cast.items():
            f = out / theme / f"{name}.svg"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(person(theme, d), encoding="utf-8")
    for name, d in ANIMALS.items():
        f = out / "ANIMALS" / f"{name}.svg"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(animal(d), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out_svg"))
