"""HTTP Range 요청으로 원격 zip(분할 zip 포함)에서 필요한 파일만 꺼낸다.

FSD50K(수십 GB, z01~zNN 분할)와 FMA(22GB)에서 고른 클립 수백~수천 개만 받으려고 쓴다.
"""

from __future__ import annotations

import bz2
import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    name: str
    disk: int
    offset: int
    method: int
    compressed: int
    size: int


def _get(url: str, start: int, end: int, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return bytes(r.read())
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2**i)
    raise RuntimeError("unreachable")


def _length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class RemoteZip:
    """parts는 디스크 순서의 주소 목록이다 (분할 zip이면 z01, z02, ..., 마지막이 .zip)."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts
        self.lengths = [_length(u) for u in parts]
        self.entries = self._read_directory()

    def _read_directory(self) -> dict[str, Entry]:
        last, n = self.parts[-1], self.lengths[-1]
        tail = _get(last, max(0, n - 65_600), n - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise ValueError("EOCD를 찾지 못했다")
        (_, _, cd_disk, _, total, cd_size, cd_offset, _) = struct.unpack(
            "<IHHHHIIH", tail[eocd : eocd + 22]
        )
        loc = tail.rfind(b"PK\x06\x07", 0, eocd)
        if loc >= 0:  # zip64
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc : loc + 20])
            rec = _get(self.parts[z64_disk], z64_off, z64_off + 55)
            (_, _, _, _, _, cd_disk, _, total, cd_size, cd_offset) = struct.unpack(
                "<IQHHIIQQQQ", rec[:56]
            )
        data = self._read_span(cd_disk, cd_offset, cd_size)
        out: dict[str, Entry] = {}
        p = 0
        for _ in range(total):
            (sig, _, _, _, method, _, _, _, comp, size, nlen, xlen, clen, disk, _, _, off) = (
                struct.unpack("<IHHHHHHIIIHHHHHII", data[p : p + 46])
            )
            if sig != 0x02014B50:
                raise ValueError("중앙 디렉터리 형식이 다르다")
            name = data[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
            extra = data[p + 46 + nlen : p + 46 + nlen + xlen]
            comp, size, off, disk = _zip64(extra, comp, size, off, disk)
            out[name] = Entry(name, disk, off, method, comp, size)
            p += 46 + nlen + xlen + clen
        return out

    def _read_span(self, disk: int, offset: int, length: int) -> bytes:
        """디스크 경계를 넘을 수 있는 구간을 읽는다."""
        buf = bytearray()
        while length > 0:
            avail = self.lengths[disk] - offset
            take = min(avail, length)
            if take > 0:
                buf += _get(self.parts[disk], offset, offset + take - 1)
            length -= take
            disk, offset = disk + 1, 0
        return bytes(buf)

    def read(self, name: str) -> bytes:
        e = self.entries[name]
        head = self._read_span(e.disk, e.offset, 30)
        nlen, xlen = struct.unpack("<HH", head[26:30])
        start = e.offset + 30 + nlen + xlen
        disk = e.disk
        while start >= self.lengths[disk]:
            start -= self.lengths[disk]
            disk += 1
        raw = self._read_span(disk, start, e.compressed)
        if e.method == 0:
            return raw
        if e.method == 8:
            return zlib.decompress(raw, -15)
        if e.method == 12:
            return bz2.decompress(raw)
        raise ValueError(f"압축 방식 {e.method}는 읽지 못한다")


def _zip64(extra: bytes, comp: int, size: int, off: int, disk: int) -> tuple[int, int, int, int]:
    p = 0
    while p + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[p : p + 4])
        if hid == 1:
            body, q = extra[p + 4 : p + 4 + hlen], 0
            if size == 0xFFFFFFFF:
                size = struct.unpack("<Q", body[q : q + 8])[0]
                q += 8
            if comp == 0xFFFFFFFF:
                comp = struct.unpack("<Q", body[q : q + 8])[0]
                q += 8
            if off == 0xFFFFFFFF:
                off = struct.unpack("<Q", body[q : q + 8])[0]
                q += 8
            if disk == 0xFFFF:
                disk = struct.unpack("<I", body[q : q + 4])[0]
        p += 4 + hlen
    return comp, size, off, disk
