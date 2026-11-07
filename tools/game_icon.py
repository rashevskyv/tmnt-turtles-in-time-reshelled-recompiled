#!/usr/bin/env python3
"""NRO icon from the player's game files: the logo of banner.png (420x95, sharp) over the blurred title
thumbnail title_icon.png (64x64: scaled to 256 alone it was blurry on hbmenu, 0.46), 256x256 JPEG.

    python3 tools/game_icon.py <game folder> <out.jpg>

Game art: write it outside git (switch/build.sh uses $TMNT_ROOT/icon/).
"""
import sys

from PIL import Image, ImageDraw, ImageFilter

game, out = sys.argv[1:3]
icon = Image.open(f'{game}/title_icon.png').convert('RGB').resize((256, 256), Image.LANCZOS)
icon = icon.filter(ImageFilter.GaussianBlur(3))
logo = Image.open(f'{game}/banner.png').convert('RGB').crop((0, 0, 176, 95))  # the logo; the turtles are right of it
w = 250
h = round(logo.height * w / logo.width)
logo = logo.resize((w, h), Image.LANCZOS)
mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(mask).rounded_rectangle((10, 8, w - 11, h - 9), 24, fill=255)
icon.paste(logo, ((256 - w) // 2, (256 - h) // 2), mask.filter(ImageFilter.GaussianBlur(8)))
icon.save(out, quality=92, subsampling=0)
print(f'{out}: banner logo over the title thumbnail, 256x256')
