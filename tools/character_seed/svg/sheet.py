"""캐릭터별 시트: 행=테마, 열=표정. python sheet.py <cast 폴더> <id...>"""
import sys
from pathlib import Path
from PIL import Image
T=["DAILY","ANIMALS","WATERCOLOR","PICTURE_BOOK","NEWSPAPER","JOSEON","WUXIA"]
E=["neutral","happy","sad","angry","surprised","embarrassed"]
root=Path(sys.argv[1]); cell=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 160
ids=[a for a in sys.argv[2:] if not a.isdigit()]
for cid in ids:
    im=Image.new("RGB",(len(E)*cell,len(T)*cell),"white")
    for r,t in enumerate(T):
        for c,e in enumerate(E):
            f=root/"png"/cid/f"{t}_{e}.png"
            if f.exists(): im.paste(Image.open(f).convert("RGB").resize((cell,cell)),(c*cell,r*cell))
    im.save(root/f"sheet_{cid}.png")
