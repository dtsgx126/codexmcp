"""image_generate 工具的纯函数单元测试 + 可选 live 集成测试。

单元测试:不启动 codex 子进程,仅验证辅助函数。
集成测试:需 env `MANJU_CODEX_IMAGE_LIVE=1` 才跑,真调 codex 出一张图。
"""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from codexmcp.server import (
    _build_imagegen_prompt,
    _locate_image_output,
    _read_png_size,
)


def _write_minimal_png(path: Path, width: int, height: int) -> None:
    """写一个最小合法 PNG(IHDR + IDAT + IEND),供 _read_png_size 读取。"""
    # IHDR chunk: 4 bytes length + "IHDR" + 13 bytes data + 4 bytes CRC
    ihdr_data = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"  # bit depth=8, color=2(RGB), 其他默认
    )
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data).to_bytes(4, "big")
    # IDAT chunk: 最小压缩数据
    idat_raw = b"\x00" + b"\x00\x00\x00" * width  # 一行透明像素
    idat_data = zlib.compress(idat_raw * height)
    idat_crc = zlib.crc32(b"IDAT" + idat_data).to_bytes(4, "big")
    # IEND chunk
    iend_crc = zlib.crc32(b"IEND").to_bytes(4, "big")

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + ihdr_data
        + ihdr_crc
        + struct.pack(">I", len(idat_data))
        + b"IDAT"
        + idat_data
        + idat_crc
        + struct.pack(">I", 0)
        + b"IEND"
        + iend_crc
    )


# ============================================================================
# _build_imagegen_prompt
# ============================================================================

class TestBuildPrompt:
    """验证 prompt 模板拼装。"""

    def test_includes_dollar_imagegen_prefix(self):
        out = _build_imagegen_prompt("画一只狐狸", "/tmp/a.png", "1:1")
        assert out.startswith("$imagegen "), "必须以 $imagegen 开头才触发技能"

    def test_includes_user_prompt(self):
        out = _build_imagegen_prompt("画一只狐狸", "/tmp/a.png", "1:1")
        assert "画一只狐狸" in out

    def test_includes_output_path(self):
        out = _build_imagegen_prompt("x", "/abs/path/to/image.png", "1:1")
        assert "/abs/path/to/image.png" in out

    def test_includes_aspect_ratio(self):
        out = _build_imagegen_prompt("x", "/tmp/a.png", "16:9")
        assert "16:9" in out

    def test_includes_silent_contract_markers(self):
        """模板必须硬约束 codex 只吐 GENERATED_OK / FAIL,否则不是静默模式。"""
        out = _build_imagegen_prompt("x", "/tmp/a.png", "1:1")
        assert "GENERATED_OK" in out
        assert "GENERATED_FAIL" in out


# ============================================================================
# _read_png_size
# ============================================================================

class TestReadPngSize:
    """验证 PNG 头尺寸读取,不依赖 Pillow。"""

    def test_reads_1024_square(self, tmp_path):
        png = tmp_path / "a.png"
        _write_minimal_png(png, 1024, 1024)
        assert _read_png_size(str(png)) == (1024, 1024)

    def test_reads_non_square(self, tmp_path):
        png = tmp_path / "b.png"
        _write_minimal_png(png, 1536, 864)
        assert _read_png_size(str(png)) == (1536, 864)

    def test_non_png_returns_none(self, tmp_path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"not a real png file" * 10)
        assert _read_png_size(str(fake)) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert _read_png_size(str(tmp_path / "nonexistent.png")) is None

    def test_truncated_header_returns_none(self, tmp_path):
        short = tmp_path / "short.png"
        short.write_bytes(b"\x89PNG\r\n\x1a\n")  # 只有 8 字节 magic,不够 24
        assert _read_png_size(str(short)) is None


# ============================================================================
# _locate_image_output
# ============================================================================

class TestLocateOutput:
    """验证输出文件定位 + fallback 逻辑。"""

    def test_returns_path_when_file_exists_and_non_trivial(self, tmp_path):
        out = tmp_path / "out.png"
        _write_minimal_png(out, 1024, 1024)  # 远大于 1024 字节阈值
        result = _locate_image_output(str(out), session_id="fake-session")
        assert result == str(out)

    def test_returns_none_when_file_missing_and_no_session(self, tmp_path):
        out = tmp_path / "out.png"
        result = _locate_image_output(str(out), session_id=None)
        assert result is None

    def test_returns_none_when_file_missing_and_no_fallback_dir(self, tmp_path):
        out = tmp_path / "out.png"
        # session_id 对应的目录不存在
        result = _locate_image_output(str(out), session_id="nonexistent-xyz-123")
        assert result is None

    def test_too_small_file_triggers_fallback(self, tmp_path, monkeypatch):
        """预期路径存在但是太小(<1024 字节)时,应当走 fallback 而不是直接返回。"""
        out = tmp_path / "out.png"
        out.write_bytes(b"tiny")  # 仅 4 字节,远低于阈值
        # fallback 目录没有候选 → 返回 None
        # 模拟 Path.home() 指向临时目录,避免触碰真实用户目录
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
        result = _locate_image_output(str(out), session_id="sess-123")
        assert result is None

    def test_fallback_copies_from_default_dir(self, tmp_path, monkeypatch):
        """预期路径不存在,但 fallback 目录里有 ig_*.png 就复制过去。"""
        out = tmp_path / "subdir" / "out.png"  # 父目录不存在,验证 mkdir
        fake_home = tmp_path / "fake_home"
        session = "sess-abc"
        fallback_dir = fake_home / ".codex" / "generated_images" / session
        fallback_dir.mkdir(parents=True)
        fallback_png = fallback_dir / "ig_0001.png"
        _write_minimal_png(fallback_png, 1254, 1254)

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        result = _locate_image_output(str(out), session_id=session)

        assert result == str(out)
        assert out.exists()
        assert _read_png_size(str(out)) == (1254, 1254)


# ============================================================================
# 集成测试(live,默认 skip)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("MANJU_CODEX_IMAGE_LIVE") != "1",
    reason="需要 MANJU_CODEX_IMAGE_LIVE=1 才跑,会消耗 Codex 订阅额度且耗时 ~30s",
)
def test_live_generate_single_image(tmp_path):
    """真调 codex,验证 image_generate tool 端到端可用。"""
    import asyncio

    from codexmcp.server import image_generate

    out = tmp_path / "live_fox.png"
    # FastMCP 的 @mcp.tool 装饰器返回原 coroutine function,可直接 await
    result = asyncio.run(
        image_generate(
            prompt="一只红狐坐在雪地上,写实摄影,50mm f/1.4",
            output_path=str(out),
            cd=tmp_path,
        )
    )

    assert result["success"] is True, f"image_generate 失败: {result.get('error')}"
    assert out.exists(), "目标文件未落盘"
    assert out.stat().st_size > 10_000, "文件太小,可能不是合法 PNG"
    assert result["original_size"] is not None
