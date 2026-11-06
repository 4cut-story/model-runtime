"""테마별 직업 캐릭터 (테마마다 4명). characters.py의 그리기 부품에 직업 옷·모자·소품을 더한다.

  python jobs.py out_svg/   → out_svg/<THEME>/<이름>.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import characters as C
from characters import CX, CY, TORSO, ROBE

# ---------- 옷 ----------


def doctor_coat(p, c, skin):
    s = p.st()
    return (f'<path d="{TORSO}" fill="{p.c("sky blue")}" {s}/>'
            f'<path d="M 350 476 L {CX} 540 L 418 476 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M {CX} 500 L 372 520 L {CX} 620 L 396 520 Z" fill="{p.c("navy")}" {p.st(3)}/>'
            f'<path d="M 186 768 L 196 580 Q 206 492 300 474 L 346 474 L 360 768 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M 582 768 L 572 580 Q 562 492 468 474 L 422 474 L 408 768 Z" fill="{p.c("white")}" {s}/>'
            f'<rect x="440" y="600" width="60" height="50" rx="4" fill="none" {p.st(4)}/>'
            f'<line x1="455" y1="590" x2="455" y2="620" stroke="{p.c("blue")}" stroke-width="6"/>')


def chef_jacket(p, c, skin):
    s = p.st()
    btn = "".join(f'<circle cx="{x}" cy="{y}" r="7" fill="{p.c("#bbbbbb")}" {p.st(2)}/>' for y in (560, 620, 680) for x in (350, 418))
    return (f'<path d="{TORSO}" fill="{p.c("white")}" {s}/>'
            f'<path d="M 330 476 L 460 476 L 470 520 L 320 520 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M {CX + 40} 520 L {CX + 40} 768" fill="none" {p.st(3)}/>' + btn +
            f'<path d="M 340 476 L 350 500 L 418 500 L 428 476" fill="{p.c("red")}" {p.st(3)}/>')


def fire_coat(p, c, skin):
    s = p.st()
    coat = p.c("#c9a46a")
    stripe = "".join(f'<rect x="196" y="{y}" width="376" height="20" fill="{p.c("yellow")}" {p.st(3)}/>'
                     f'<rect x="196" y="{y + 6}" width="376" height="8" fill="#e8e8e8"/>' for y in (620, 700))
    return (f'<path d="{TORSO}" fill="{coat}" {s}/>' + stripe +
            f'<path d="M 320 470 L {CX} 540 L 448 470 L 448 500 L {CX} 570 L 320 500 Z" fill="{coat}" {s}/>'
            f'<line x1="{CX}" y1="540" x2="{CX}" y2="768" {s}/>')


def police(p, c, skin):
    s = p.st()
    return (f'<path d="{TORSO}" fill="{p.c("navy")}" {s}/>'
            f'<path d="M 330 470 L {CX} 520 L 360 560 L 310 490 Z M 438 470 L {CX} 520 L 408 560 L 458 490 Z" fill="{p.c("navy")}" {s}/>'
            f'<path d="M {CX} 520 L 372 540 L {CX} 640 L 396 540 Z" fill="{p.c("black")}" {p.st(3)}/>'
            f'<path d="M 230 560 L 290 560 L 290 610 L 260 630 L 230 610 Z" fill="{p.c("#c9a227")}" {p.st(3)}/>'
            f'<rect x="440" y="590" width="80" height="20" rx="3" fill="{p.c("white")}" {p.st(3)}/>'
            f'<rect x="186" y="490" width="80" height="14" fill="{p.c("#c9a227")}" opacity="0.8" transform="rotate(12 226 497)"/>')


def apron_over(color):
    def f(p, c, skin):
        s = p.st()
        return (f'<path d="{TORSO}" fill="{c}" {s}/>'
                f'<path d="M 330 476 Q {CX} 520 438 476" fill="none" {p.st(8)}/>'
                f'<path d="M 316 560 L 452 560 L 476 768 L 292 768 Z" fill="{p.c(color)}" {s}/>'
                f'<path d="M 316 560 L 330 482 M 452 560 L 438 482" fill="none" stroke="{p.c(color)}" stroke-width="12"/>'
                f'<rect x="350" y="640" width="68" height="46" rx="5" fill="none" {p.st(4)}/>')
    return f


def postal(p, c, skin):
    s = p.st()
    return (C.clothes(p, "shirt", "blue", skin) +
            f'<path d="M 300 476 L 560 768" stroke="{p.c("dark brown")}" stroke-width="26" fill="none"/>'
            f'<path d="M 300 476 L 560 768" {p.st(3)} fill="none"/>'
            f'<rect x="200" y="660" width="120" height="100" rx="10" fill="{p.c("brown")}" {s}/>'
            f'<path d="M 200 690 L 320 690" {p.st(3)}/>')


def trench(p, c, skin):
    s = p.st()
    tc = p.c("#bfa57a")
    return (f'<path d="{TORSO}" fill="{tc}" {s}/>'
            f'<path d="M 340 476 L {CX} 540 L 428 476 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M 300 466 L 360 620 L 320 640 L 270 500 Z M 468 466 L 408 620 L 448 640 L 498 500 Z" fill="{tc}" {s}/>'
            f'<line x1="{CX}" y1="540" x2="{CX}" y2="768" {s}/>'
            + "".join(f'<circle cx="{x}" cy="{y}" r="8" fill="{p.c("dark brown")}" {p.st(2)}/>' for y in (600, 680) for x in (350, 418))
            + f'<rect x="196" y="700" width="376" height="26" fill="{tc}" {s}/>')


def coverall(p, c, skin):
    s = p.st()
    return (f'<path d="{TORSO}" fill="{c}" {s}/>'
            f'<path d="M 330 470 L {CX} 520 L 360 560 L 310 490 Z M 438 470 L {CX} 520 L 408 560 L 458 490 Z" fill="{c}" {s}/>'
            f'<line x1="{CX}" y1="520" x2="{CX}" y2="768" {s}/>'
            f'<rect x="420" y="580" width="70" height="60" rx="4" fill="none" {p.st(4)}/>'
            f'<path d="M 230 650 Q 260 640 290 660" fill="none" stroke="#111" stroke-width="10" opacity="0.5"/>')


def waitress(p, c, skin):
    s = p.st()
    return (C.clothes(p, "dress", "gray", skin) +
            f'<path d="M 300 590 L 468 590 L 488 768 L 280 768 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M 300 590 Q 280 560 260 590 M 468 590 Q 488 560 508 590" fill="none" {p.st(4)}/>'
            + "".join(f'<path d="M {x} 768 q 12 -16 24 0" fill="none" {p.st(3)}/>' for x in range(284, 480, 26)))


def astronaut(p, c, skin):
    s = p.st()
    return (f'<path d="M 170 768 L 180 580 Q 190 480 300 466 L 468 466 Q 578 480 588 580 L 598 768 Z" fill="{p.c("white")}" {s}/>'
            f'<rect x="320" y="580" width="128" height="90" rx="12" fill="{p.c("#bbbbbb")}" {s}/>'
            + "".join(f'<circle cx="{350 + i * 34}" cy="625" r="10" fill="{p.c(col)}" {p.st(2)}/>' for i, col in enumerate(("red", "yellow", "blue")))
            + f'<rect x="220" y="560" width="60" height="40" fill="{p.c("blue")}" {p.st(3)}/>'
            f'<path d="M 180 640 L 260 640 M 508 640 L 588 640" {p.st(4)}/>')


def podol(p, c, skin):  # 포졸: 남색 저고리 위 붉은 쾌자, 가슴에 원
    s = p.st()
    return (C.clothes(p, "hanbok_m", "indigo", skin) +
            f'<path d="M 250 768 L 260 560 Q 280 500 330 480 L {CX} 600 L 438 480 Q 488 500 508 560 L 518 768 Z" fill="{p.c("crimson")}" {s}/>'
            f'<circle cx="{CX}" cy="680" r="48" fill="{p.c("white")}" {s}/>'
            f'<path d="M {CX - 22} 660 L {CX + 22} 660 M {CX} 650 L {CX} 712 M {CX - 24} 690 L {CX + 24} 690" {p.st(5)}/>')


def peddler(p, c, skin):  # 보부상: 지게 멜빵
    s = p.st()
    return (C.clothes(p, "hanbok_m", "#b8a986", skin) +
            f'<path d="M 280 480 L 300 768 M 488 480 L 468 768" fill="none" stroke="{p.c("dark brown")}" stroke-width="20"/>'
            f'<path d="M 230 300 L 260 768 M 538 300 L 508 768" fill="none" stroke="{p.c("brown")}" stroke-width="22"/>'
            f'<path d="M 230 300 L 260 768 M 538 300 L 508 768" fill="none" {p.st(3)}/>')


def jumo(p, c, skin):  # 주모: 한복 + 행주치마
    s = p.st()
    return (C.clothes(p, "hanbok_w", "#8a6a4a", skin).replace(p.c("cream"), p.c("jade")) +
            f'<path d="M 280 620 L 488 620 L 500 768 L 268 768 Z" fill="{p.c("white")}" {s}/>')


def uinyeo(p, c, skin):  # 의녀: 연옥색 저고리, 남색 치마
    s = p.st()
    return C.clothes(p, "hanbok_w", "navy", skin).replace(p.c("cream"), p.c("#cfe6e0"))


def monk(p, c, skin):  # 소림 무승: 주황 가사, 한쪽 어깨
    s = p.st()
    return (f'<path d="{ROBE}" fill="{p.c("#e79a3c")}" {s}/>'
            f'<path d="M 160 580 Q 176 490 296 470 L 560 768 L 140 768 Z" fill="{p.c("#c0552d")}" {s}/>'
            + "".join(f'<circle cx="{CX + 90 * math.cos(a / 8)}" cy="{500 + 70 * math.sin(a / 8)}" r="13" fill="{p.c("dark brown")}" {p.st(2)}/>'
                      for a in range(2, 25, 2)))


def innkeeper(p, c, skin):
    s = p.st()
    return (C.clothes(p, "robe", "#7a5a3a", skin) +
            f'<path d="M 300 640 L 468 640 L 488 768 L 280 768 Z" fill="{p.c("white")}" {s}/>')


def assassin(p, c, skin):
    s = p.st()
    return (f'<path d="M 250 300 C 240 150 330 120 {CX} 120 C 438 120 528 150 518 300 L 540 500 L 228 500 Z" fill="{p.c("black")}" {s}/>'
            + C.clothes(p, "robe", "black", skin))


for k, fn in {"doctor": doctor_coat, "chef": chef_jacket, "fire": fire_coat, "police": police,
              "florist": apron_over("green"), "barista": apron_over("brown"), "painter": apron_over("cream"),
              "postal": postal, "trench": trench, "coverall": coverall, "waitress": waitress,
              "astronaut": astronaut, "podol": podol, "peddler": peddler, "jumo": jumo, "uinyeo": uinyeo,
              "monk": monk, "innkeeper": innkeeper, "assassin": assassin}.items():
    C.EXTRA_CLOTHES[k] = fn

# ---------- 모자·소품 ----------


def A(fn):
    C.EXTRA_ACC[fn.__name__.replace("_", " ")] = fn
    return fn


@A
def stethoscope(p, skin):
    return (f'<path d="M 330 480 Q 320 600 {CX} 620 Q 448 600 438 480" fill="none" stroke="#555" stroke-width="7"/>'
            f'<circle cx="{CX}" cy="640" r="18" fill="#cfd6db" {p.st(4)}/>')


@A
def chef_hat(p, skin):
    s = p.st()
    return (f'<rect x="286" y="150" width="196" height="60" fill="{p.c("white")}" {s}/>'
            f'<path d="M 290 160 C 220 140 240 40 310 60 C 320 0 448 0 458 60 C 528 40 548 140 478 160 Z" fill="{p.c("white")}" {s}/>')


@A
def fire_helmet(p, skin):
    s = p.st()
    return (f'<path d="M 230 230 Q 230 110 {CX} 100 Q 538 110 538 230 Q 600 236 600 250 L 168 250 Q 168 236 230 230 Z" fill="{p.c("red")}" {s}/>'
            f'<path d="M {CX} 100 L {CX} 230" {p.st(5)}/>'
            f'<path d="M 350 150 L 418 150 L 418 210 L {CX} 225 L 350 210 Z" fill="{p.c("#c9a227")}" {p.st(3)}/>')


@A
def police_cap(p, skin):
    s = p.st()
    return (f'<path d="M 240 200 Q 250 110 {CX} 108 Q 518 110 528 200 Z" fill="{p.c("navy")}" {s}/>'
            f'<rect x="250" y="196" width="268" height="30" fill="{p.c("black")}" {s}/>'
            f'<path d="M 270 226 Q {CX} 270 498 226 Z" fill="{p.c("black")}" {s}/>'
            f'<circle cx="{CX}" cy="160" r="18" fill="{p.c("#c9a227")}" {p.st(3)}/>')


@A
def cap(p, skin):
    s = p.st()
    return (f'<path d="M 250 220 Q 250 120 {CX} 118 Q 518 120 518 220 Z" fill="{p.c("blue")}" {s}/>'
            f'<path d="M 250 220 Q {CX} 200 518 220 Q 600 230 590 250 Q {CX} 228 250 240 Z" fill="{p.c("blue")}" {s}/>')


@A
def mech_cap(p, skin):
    return cap(p, skin).replace(p.c("blue"), p.c("#555555"))


@A
def fedora(p, skin):
    s = p.st()
    return (f'<ellipse cx="{CX}" cy="210" rx="190" ry="34" fill="{p.c("#444444")}" {s}/>'
            f'<path d="M 290 210 L 300 120 Q {CX} 90 468 120 L 478 210 Z" fill="{p.c("#444444")}" {s}/>'
            f'<rect x="292" y="176" width="184" height="22" fill="{p.c("#111111")}"/>'
            f'<rect x="420" y="150" width="40" height="26" fill="{p.c("white")}" {p.st(2)}/>')


@A
def headband_white(p, skin):
    return f'<path d="M 262 220 Q {CX} 170 506 220 L 506 245 Q {CX} 195 262 245 Z" fill="{p.c("white")}" {p.st(3)}/>'


@A
def space_helmet(p, skin):
    return (f'<circle cx="{CX}" cy="{CY + 10}" r="175" fill="#bfe3ff" fill-opacity="0.18" {p.st(6)}/>'
            f'<path d="M 270 160 Q 300 130 340 125" fill="none" stroke="#fff" stroke-width="10" stroke-linecap="round" opacity="0.8"/>'
            f'<path d="M 230 440 Q {CX} 500 538 440 L 548 480 Q {CX} 540 220 480 Z" fill="{p.c("#bbbbbb")}" {p.st(4)}/>')


@A
def bouquet(p, skin):
    flowers = "".join(f'<circle cx="{x}" cy="{y}" r="24" fill="{p.c(col)}" {p.st(3)}/><circle cx="{x}" cy="{y}" r="8" fill="{p.c("yellow")}"/>'
                      for x, y, col in ((230, 600, "pink"), (270, 570, "red"), (300, 610, "purple"), (255, 630, "white")))
    return (f'<path d="M 250 640 L 280 760 L 320 760 L 300 640 Z" fill="{p.c("green")}" {p.st(3)}/>' + flowers)


@A
def coffee(p, skin):
    return (f'<path d="M 230 640 L 300 640 L 290 740 L 240 740 Z" fill="{p.c("white")}" {p.st(4)}/>'
            f'<rect x="232" y="670" width="66" height="24" fill="{p.c("brown")}"/>'
            f'<path d="M 250 620 q 10 -20 0 -40 M 275 620 q 10 -20 0 -40" fill="none" {p.st(3)} opacity="0.6"/>')


@A
def palette(p, skin):
    blobs = "".join(f'<circle cx="{x}" cy="{y}" r="12" fill="{p.c(col)}"/>' for x, y, col in
                    ((220, 650, "red"), (250, 630, "yellow"), (285, 640, "blue"), (300, 675, "green")))
    return (f'<path d="M 200 660 C 200 600 330 590 330 650 C 330 700 280 690 270 720 C 250 760 200 720 200 660 Z" fill="{p.c("#e8d3a8")}" {p.st(4)}/>'
            + blobs + f'<line x1="470" y1="760" x2="540" y2="560" stroke="{p.c("brown")}" stroke-width="10"/><circle cx="540" cy="556" r="10" fill="{p.c("blue")}"/>')


@A
def beret_red(p, skin):
    return C.accessory(p, "beret", skin).replace(p.c("#333"), p.c("red"))


@A
def notepad(p, skin):
    return (f'<rect x="210" y="600" width="100" height="130" rx="6" fill="{p.c("white")}" {p.st(4)}/>'
            + "".join(f'<line x1="224" y1="{y}" x2="296" y2="{y}" {p.st(2)}/>' for y in (630, 655, 680, 705))
            + f'<line x1="330" y1="740" x2="380" y2="610" stroke="{p.c("yellow")}" stroke-width="12"/>'
            f'<line x1="330" y1="740" x2="380" y2="610" {p.st(2)}/>')


@A
def wrench(p, skin):
    return (f'<line x1="220" y1="760" x2="290" y2="600" stroke="#aaa" stroke-width="22" stroke-linecap="round"/>'
            f'<line x1="220" y1="760" x2="290" y2="600" {p.st(3)}/>'
            f'<path d="M 270 600 a 30 30 0 1 1 46 10 l -20 -8 l -6 -20 Z" fill="#aaa" {p.st(3)}/>')


@A
def tray(p, skin):
    return (f'<ellipse cx="540" cy="600" rx="90" ry="18" fill="#cccccc" {p.st(4)}/>'
            f'<path d="M 510 540 L 560 540 L 555 594 L 515 594 Z" fill="{p.c("white")}" {p.st(3)}/>')


@A
def jeollip(p, skin):  # 포졸 전립: 검은 벙거지 + 붉은 술
    s = p.st()
    return (f'<ellipse cx="{CX}" cy="205" rx="200" ry="36" fill="{p.c("black")}" {s}/>'
            f'<path d="M 300 205 Q 300 100 {CX} 96 Q 468 100 468 205 Z" fill="{p.c("black")}" {s}/>'
            f'<path d="M {CX} 96 L {CX} 70" {p.st(5)}/><circle cx="{CX}" cy="66" r="18" fill="{p.c("red")}" {p.st(3)}/>'
            f'<path d="M 440 120 Q 500 160 520 230" fill="none" stroke="{p.c("red")}" stroke-width="10"/>')


@A
def paeraengi(p, skin):  # 보부상 패랭이(대나무 갓) + 목화송이
    s = p.st()
    return (f'<ellipse cx="{CX}" cy="205" rx="200" ry="36" fill="{p.c("#d4b77a")}" {s}/>'
            f'<path d="M 310 205 L 320 115 Q {CX} 100 448 115 L 458 205 Z" fill="{p.c("#d4b77a")}" {s}/>'
            + "".join(f'<circle cx="{x}" cy="126" r="14" fill="#fff" {p.st(2)}/>' for x in (420, 446)))


@A
def jumo_scarf(p, skin):
    return (f'<path d="M 250 260 C 240 140 330 120 {CX} 120 C 438 120 528 140 518 260 Q {CX} 200 250 260 Z" fill="{p.c("white")}" {p.st()}/>'
            f'<path d="M 500 200 L 560 260 L 520 280 Z" fill="{p.c("white")}" {p.st(3)}/>')


@A
def garima(p, skin):  # 의녀 가리마: 머리 위 검은 천
    return (f'<path d="M 280 200 Q {CX} 110 488 200 L 500 280 Q {CX} 230 268 280 Z" fill="{p.c("black")}" {p.st()}/>')


@A
def herb_basket(p, skin):
    return (f'<path d="M 200 640 L 320 640 L 305 740 L 215 740 Z" fill="{p.c("#c9a46a")}" {p.st(4)}/>'
            + "".join(f'<path d="M {x} 640 q 10 -40 30 -50" fill="none" stroke="{p.c("green")}" stroke-width="8"/>' for x in (220, 250, 280)))


@A
def bald(p, skin):
    return "".join(f'<circle cx="{CX + dx}" cy="{196}" r="6" fill="#555"/>' for dx in (-40, -14, 14, 40)) + \
        "".join(f'<circle cx="{CX + dx}" cy="{224}" r="6" fill="#555"/>' for dx in (-27, 0, 27))


@A
def abacus(p, skin):
    rows = "".join(f'<line x1="200" y1="{y}" x2="330" y2="{y}" {p.st(2)}/>' + "".join(
        f'<ellipse cx="{x}" cy="{y}" rx="8" ry="6" fill="{p.c("brown")}"/>' for x in (215, 232, 270, 300)) for y in (630, 660, 690))
    return f'<rect x="192" y="610" width="146" height="100" fill="none" {p.st(5)}/>' + rows


@A
def shop_cap(p, skin):
    return f'<path d="M 280 200 Q 280 120 {CX} 116 Q 488 120 488 200 Z" fill="{p.c("black")}" {p.st()}/>'


@A
def gourd(p, skin):
    return (f'<circle cx="250" cy="700" r="40" fill="{p.c("#d4a23a")}" {p.st(4)}/>'
            f'<circle cx="250" cy="640" r="26" fill="{p.c("#d4a23a")}" {p.st(4)}/>'
            f'<rect x="240" y="604" width="20" height="14" fill="{p.c("red")}" {p.st(2)}/>')


@A
def herb_hat(p, skin):  # 신의: 망사 두건
    return f'<path d="M 290 190 Q {CX} 90 478 190 L 470 220 Q {CX} 170 298 220 Z" fill="{p.c("#4b6b5a")}" {p.st()}/>'


@A
def mask(p, skin):
    return (f'<path d="M 264 350 Q {CX} 335 504 350 L 500 430 Q {CX} 480 268 430 Z" fill="{p.c("black")}" {p.st()}/>')


@A
def dagger(p, skin):
    return (f'<path d="M 530 760 L 560 580 L 576 760 Z" fill="#cfd6db" {p.st(3)}/>'
            f'<line x1="530" y1="700" x2="590" y2="704" {p.st(10)}/>')


# 동물용: 작은 머리 위치에 맞춘 모자는 사람과 같은 좌표를 쓴다 (머리가 비슷한 크기)

CAST = {
    "DAILY": {
        "도현": dict(hair="short", hair_color="black", clothes="doctor", color="white", eyes="round", skin="warm", acc=["stethoscope"]),
        "강민": dict(hair="short", hair_color="brown", clothes="chef", color="white", eyes="smile", mouth="open", skin="tan", acc=["chef hat"]),
        "태산": dict(hair="short", hair_color="black", clothes="fire", color="tan", eyes="round", mouth="open", skin="tan", acc=["fire helmet"]),
        "민서": dict(hair="ponytail", hair_color="black", clothes="police", color="navy", eyes="lash", mouth="flat", skin="light", acc=["police cap"]),
    },
    "WATERCOLOR": {
        "봄비": dict(hair="long", hair_color="brown", clothes="florist", color="cream", eyes="smile", skin="light", acc=["bouquet"]),
        "나래": dict(hair="bob", hair_color="black", clothes="barista", color="white", eyes="round", skin="warm", acc=["coffee"]),
        "하늘": dict(hair="wavy", hair_color="blonde", clothes="painter", color="sky blue", eyes="lash", skin="light", acc=["beret red", "palette"]),
        "우진": dict(hair="short", hair_color="dark brown", clothes="postal", color="blue", eyes="dot", skin="warm", acc=["cap"]),
    },
    "PICTURE_BOOK": {
        "불꽃이": dict(hair="short", hair_color="black", clothes="fire", color="tan", eyes="round", mouth="open", skin="warm", acc=["fire helmet"]),
        "튼튼이": dict(hair="pigtails", hair_color="brown", clothes="doctor", color="white", eyes="round", skin="light", acc=["stethoscope"]),
        "냠냠이": dict(hair="curly", hair_color="black", clothes="chef", color="white", eyes="smile", mouth="open", skin="tan", acc=["chef hat"]),
        "별이": dict(hair="bob", hair_color="black", clothes="astronaut", color="white", eyes="round", mouth="open", skin="light", acc=["space helmet"]),
    },
    "NEWSPAPER": {
        "박기자": dict(hair="short", hair_color="black", clothes="shirt", color="white", eyes="round", acc=["fedora", "notepad"]),
        "강형사": dict(hair="short", hair_color="black", clothes="trench", color="tan", eyes="sharp", mouth="flat", acc=["fedora"]),
        "덕배": dict(hair="short", hair_color="black", clothes="coverall", color="gray", eyes="smile", mouth="open", acc=["mech cap", "wrench"]),
        "순이": dict(hair="ponytail", hair_color="black", clothes="waitress", color="gray", eyes="lash", acc=["headband white", "tray"]),
    },
    "JOSEON": {
        "돌쇠": dict(hair="topknot", hair_color="black", clothes="podol", color="crimson", eyes="dot", mouth="flat", skin="tan", acc=["jeollip"]),
        "만복": dict(hair="topknot", hair_color="black", clothes="peddler", color="cream", eyes="smile", mouth="open", skin="tan", acc=["paeraengi"]),
        "주모": dict(hair="chignon", hair_color="black", clothes="jumo", color="brown", eyes="smile", mouth="open", skin="warm", acc=["jumo scarf"]),
        "장금": dict(hair="chignon", hair_color="black", clothes="uinyeo", color="navy", eyes="round", mouth="flat", skin="light", acc=["garima", "herb basket"]),
    },
    "WUXIA": {
        "왕장궤": dict(hair="short", hair_color="black", clothes="innkeeper", color="brown", eyes="smile", mouth="open", skin="warm", acc=["shop cap", "mustache", "abacus"]),
        "혜능": dict(hair="none", hair_color="black", clothes="monk", color="orange", eyes="round", mouth="flat", skin="tan", acc=["bald"]),
        "화타": dict(hair="wuxia_high", hair_color="gray", clothes="robe", color="jade", eyes="smile", skin="warm", acc=["herb hat", "mustache", "gourd"]),
        "흑영": dict(hair="none", hair_color="black", clothes="assassin", color="black", eyes="sharp", skin="light", acc=["mask", "dagger"]),
    },
}

ANIMALS = {
    "하양": dict(animal="rabbit", clothes="doctor", color="white", extra=["stethoscope"]),
    "뭉치": dict(animal="bear", clothes="chef", color="white", extra=["chef hat"]),
    "바둑이": dict(animal="dog", clothes="fire", color="tan", extra=["fire helmet"]),
    "냥순": dict(animal="cat", clothes="police", color="navy", extra=["police cap"]),
}


def main(out: Path) -> None:
    for theme, cast in CAST.items():
        for name, d in cast.items():
            f = out / theme / f"{name}.svg"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(C.person(theme, d), encoding="utf-8")
    for name, d in ANIMALS.items():
        f = out / "ANIMALS" / f"{name}.svg"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(C.animal(d), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out_svg"))
