"""터미널 출력 텍스트를 보고서 첨부용 고해상도 다크 터미널 PNG 이미지로 렌더링한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_mono_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_terminal_png(title: str, text: str, out_path: Path, max_cols: int = 104, max_lines: int = 44) -> None:
    font = load_mono_font(16)
    title_font = load_mono_font(15)

    # Wrap long lines cleanly
    lines: list[str] = []
    for raw in text.replace("\r", "").splitlines():
        if not raw:
            lines.append("")
            continue
        while len(raw) > max_cols:
            lines.append(raw[:max_cols])
            raw = raw[max_cols:]
        lines.append(raw)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"... ({len(lines) - max_lines + 1} lines omitted)"]
    if not lines:
        lines = ["(no output)"]

    pad_x, pad_y = 22, 18
    header_h = 38
    line_h = 22
    char_w = 10
    width = max(760, min(1200, max(len(l) for l in lines) * char_w + pad_x * 2 + 20))
    height = header_h + pad_y * 2 + len(lines) * line_h

    img = Image.new("RGB", (width, height), (24, 26, 33))
    draw = ImageDraw.Draw(img)

    # Window titlebar
    draw.rectangle([(0, 0), (width, header_h)], fill=(38, 42, 52))
    for i, col in enumerate(((237, 106, 94), (244, 191, 79), (97, 197, 84))):
        cx = 18 + i * 20
        cy = header_h // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=col)
    draw.text((88, 10), title, fill=(215, 220, 230), font=title_font)

    y = header_h + pad_y
    for line in lines:
        if line.startswith("$ ") or line.startswith("seongho06@"):
            col = (126, 231, 135)
        elif "Submitted batch job" in line or "COMPLETED" in line or "RUNNING" in line:
            col = (121, 192, 255)
        elif "ERROR" in line or "FAIL" in line:
            col = (255, 123, 114)
        else:
            col = (230, 235, 242)
        draw.text((pad_x, y), line, fill=col, font=font)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--in-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    text = args.in_file.read_text(encoding="utf-8", errors="replace")
    render_terminal_png(args.title, text, args.out)
    print(f"📸 캡처 저장: {args.out}")


if __name__ == "__main__":
    main()
