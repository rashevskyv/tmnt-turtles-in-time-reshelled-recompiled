#!/usr/bin/env python3
"""NRO icon from an STFS package: the title thumbnail (PNG, size at 0x1716, data at 0x571A), 64 -> 256 JPEG.

    python3 tools/stfs_icon.py <STFS package> <out.jpg>

Game art: write it outside git (switch/build.sh uses /root/tmnt/icon/).
"""
import io
import struct
import sys

from PIL import Image

pkg, out = sys.argv[1:3]
head = open(pkg, 'rb').read(0x9800)
size = struct.unpack_from('>I', head, 0x1716)[0]
im = Image.open(io.BytesIO(head[0x571A:0x571A + size])).convert('RGB')
im.resize((256, 256), Image.LANCZOS).save(out, quality=92, subsampling=0)
print(f'{out}: title thumbnail {im.size[0]}x{im.size[1]} -> 256x256')
