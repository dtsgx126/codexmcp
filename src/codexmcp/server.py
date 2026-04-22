"""FastMCP server implementation for the Codex MCP project."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from pydantic import BeforeValidator, Field
import shutil

mcp = FastMCP("Codex MCP Server-from guda.studio")


def _empty_str_to_none(value: str | None) -> str | None:
    """Convert empty strings to None for optional UUID parameters."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def run_shell_command(cmd: list[str]) -> Generator[str, None, None]:
    """Execute a command and stream its output line-by-line.

    Args:
        cmd: Command and arguments as a list (e.g., ["codex", "exec", "prompt"])

    Yields:
        Output lines from the command
    """
    # On Windows, codex is exposed via a *.cmd shim. Use cmd.exe with /s so
    # user prompts containing quotes/newlines aren't reinterpreted as shell syntax.
    popen_cmd = cmd.copy()
    codex_path = shutil.which('codex') or cmd[0]
    popen_cmd[0] = codex_path

    process = subprocess.Popen(
        popen_cmd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
    )

    output_queue: queue.Queue[str | None] = queue.Queue()
    GRACEFUL_SHUTDOWN_DELAY = 0.3

    def is_turn_completed(line: str) -> bool:
        """Check if the line indicates turn completion via JSON parsing."""
        try:
            data = json.loads(line)
            return data.get("type") == "turn.completed"
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    def read_output() -> None:
        """Read process output in a separate thread."""
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                stripped = line.strip()
                output_queue.put(stripped)
                if is_turn_completed(stripped):
                    time.sleep(GRACEFUL_SHUTDOWN_DELAY)
                    process.terminate()
                    break
            process.stdout.close()
        output_queue.put(None)

    thread = threading.Thread(target=read_output)
    thread.start()

    # Yield lines while process is running
    while True:
        try:
            line = output_queue.get(timeout=0.5)
            if line is None:
                break
            yield line
        except queue.Empty:
            if process.poll() is not None and not thread.is_alive():
                break

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    thread.join(timeout=5)

    while not output_queue.empty():
        try:
            line = output_queue.get_nowait()
            if line is not None:
                yield line
        except queue.Empty:
            break

def windows_escape(prompt):
    """
    Windows 风格的字符串转义函数。
    把常见特殊字符转义成 \\ 形式，适合命令行、JSON 或路径使用。
    比如：\n 变成 \\n，" 变成 \\"。
    """
    # 先处理反斜杠，避免它干扰其他替换
    result = prompt.replace('\\', '\\\\')
    # 双引号，转义成 \"，防止字符串边界乱套
    result = result.replace('"', '\\"')
    # 换行符，Windows 常用 \r\n，但我们分开转义
    result = result.replace('\n', '\\n')
    result = result.replace('\r', '\\r')
    # 制表符，空格的“超级版”
    result = result.replace('\t', '\\t')
    # 其他常见：退格符（像按了后退键）、换页符（打印机跳页用）
    result = result.replace('\b', '\\b')
    result = result.replace('\f', '\\f')
    # 如果有单引号，也转义下（不过 Windows 命令行不那么严格，但保险起见）
    result = result.replace("'", "\\'")
    
    return result

@mcp.tool(
    name="codex",
    description="""
    Executes a non-interactive Codex session via CLI to perform AI-assisted coding tasks in a secure workspace.
    This tool wraps the `codex exec` command, enabling model-driven code generation, debugging, or automation based on natural language prompts.
    It supports resuming ongoing sessions for continuity and enforces sandbox policies to prevent unsafe operations. Ideal for integrating Codex into MCP servers for agentic workflows, such as code reviews or repo modifications.

    **Key Features:**
        - **Prompt-Driven Execution:** Send task instructions to Codex for step-by-step code handling.
        - **Workspace Isolation:** Operate within a specified directory, with optional Git repo skipping.
        - **Security Controls:** Three sandbox levels balance functionality and safety.
        - **Session Persistence:** Resume prior conversations via `SESSION_ID` for iterative tasks.

    **Edge Cases & Best Practices:**
        - Ensure `cd` exists and is accessible; tool fails silently on invalid paths.
        - When `sandbox` / `yolo` are omitted, the values in ~/.codex/config.toml (sandbox_mode, approval_policy) take effect.
        - Pass `sandbox="read-only"` explicitly for review/analysis tasks that must not modify files.
        - If needed, set `return_all_messages` to `True` to parse "all_messages" for detailed tracing (e.g., reasoning, tool calls, etc.).
    """,
    meta={"version": "0.0.0", "author": "guda.studio"},
)
async def codex(
    PROMPT: Annotated[str, "Instruction for the task to send to codex."],
    cd: Annotated[Path, "Set the workspace root for codex before executing the task."],
    sandbox: Annotated[
        Optional[Literal["read-only", "workspace-write", "danger-full-access"]],
        Field(
            description="Sandbox policy. If omitted, codex CLI uses `sandbox_mode` from ~/.codex/config.toml. Pass explicitly to override."
        ),
    ] = None,
    SESSION_ID: Annotated[
        str,
        "Resume the specified session of the codex. Defaults to `None`, start a new session.",
    ] = "",
    skip_git_repo_check: Annotated[
        bool,
        "Allow codex running outside a Git repository (useful for one-off directories).",
    ] = True,
    return_all_messages: Annotated[
        bool,
        "Return all messages (e.g. reasoning, tool calls, etc.) from the codex session. Set to `False` by default, only the agent's final reply message is returned.",
    ] = False,
    image: Annotated[
        List[Path],
        Field(
            description="Attach one or more image files to the initial prompt. Separate multiple paths with commas or repeat the flag.",
        ),
    ] = [],
    model: Annotated[
        str,
        Field(
            description="The model to use for the codex session. This parameter is strictly prohibited unless explicitly specified by the user.",
        ),
    ] = "",
    yolo: Annotated[
        Optional[bool],
        Field(
            description="Skip all approvals. If omitted, codex CLI uses `approval_policy` from ~/.codex/config.toml. Pass True explicitly to force unattended execution.",
        ),
    ] = None,
    profile: Annotated[
        str,
        "Configuration profile name to load from `~/.codex/config.toml`. This parameter is strictly prohibited unless explicitly specified by the user.",
    ] = "",
) -> Dict[str, Any]:
    """Execute a Codex CLI session and return the results."""
    # Build command as list to avoid injection
    cmd = ["codex", "exec", "--cd", str(cd), "--json"]

    # sandbox/yolo 省略时不传 flag，让 codex CLI 使用 ~/.codex/config.toml 的值
    if sandbox is not None:
        cmd.extend(["--sandbox", sandbox])

    if len(image):
        # 修复 Windows 下 Path 对象拼接报错:
        # `sequence item 0: expected str instance, WindowsPath found`
        cmd.extend(["--image", ",".join(str(p) for p in image)])

    if model:
        cmd.extend(["--model", model])

    if profile:
        cmd.extend(["--profile", profile])

    if yolo is True:
        cmd.append("--yolo")
    
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    if SESSION_ID:
        cmd.extend(["resume", str(SESSION_ID)])
        
    if os.name == "nt":
        PROMPT = windows_escape(PROMPT)
    else:
        PROMPT = PROMPT
    cmd += ['--', PROMPT]

    all_messages: list[Dict[str, Any]] = []
    agent_messages = ""
    success = True
    err_message = ""
    thread_id: Optional[str] = None

    for line in run_shell_command(cmd):
        try:
            line_dict = json.loads(line.strip())
            all_messages.append(line_dict)
            item = line_dict.get("item", {})
            item_type = item.get("type", "")
            if item_type == "agent_message":
                agent_messages = agent_messages + item.get("text", "")
            if line_dict.get("thread_id") is not None:
                thread_id = line_dict.get("thread_id")
            if "fail" in line_dict.get("type", ""):
                success = False if len(agent_messages) == 0 else success
                err_message += "\n\n[codex error] " + line_dict.get("error", {}).get("message", "")
            if "error" in line_dict.get("type", ""):
                error_msg = line_dict.get("message", "")
                import re
                is_reconnecting = bool(re.match(r'^Reconnecting\.\.\.\s+\d+/\d+', error_msg))
                
                if not is_reconnecting:
                    success = False if len(agent_messages) == 0 else success
                    err_message += "\n\n[codex error] " + error_msg
                    
        except json.JSONDecodeError:
            # import sys
            # print(f"Ignored non-JSON line: {line}", file=sys.stderr)
            err_message += "\n\n[json decode error] " + line
            continue
            
        except Exception as error:
            err_message += "\n\n[unexpected error] " + f"Unexpected error: {error}. Line: {line!r}"
            success = False
            break

    if thread_id is None:
        success = False
        err_message = "Failed to get `SESSION_ID` from the codex session. \n\n" + err_message
        
    if len(agent_messages) == 0:
        success = False
        err_message = "Failed to get `agent_messages` from the codex session. \n\n You can try to set `return_all_messages` to `True` to get the full reasoning information. " + err_message

    if success:
        result: Dict[str, Any] = {
            "success": True,
            "SESSION_ID": thread_id,
            "agent_messages": agent_messages,
            # "PROMPT": PROMPT,
        }
        
    else:
        result = {"success": False, "error": err_message}
        
    if return_all_messages:
            result["all_messages"] = all_messages

    return result


# =============================================================================
# image_generate 专用工具(2026-04 新增)
# 走 codex CLI 的 `$imagegen` 技能,调用 gpt-image-2,消耗 Codex 订阅额度。
# =============================================================================

_IMAGEGEN_PROMPT_TEMPLATE = (
    "$imagegen {user_prompt}\n\n"
    "# 严格输出约定(必须遵守)\n"
    "- 生成 {aspect_ratio} 比例的图\n"
    "- 保存到: {output_path}\n"
    "- 成功仅输出一行: GENERATED_OK: {output_path}\n"
    "- 失败仅输出一行: GENERATED_FAIL: <原因>\n"
    "- 不要自评、不要解释、不要多余文字"
)


def _build_imagegen_prompt(
    user_prompt: str, output_path: str, aspect_ratio: str
) -> str:
    """拼装 `$imagegen` 的严格输出约定 prompt。"""
    return _IMAGEGEN_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        output_path=output_path,
        aspect_ratio=aspect_ratio,
    )


def _read_png_size(path: str) -> Optional[Tuple[int, int]]:
    """从 PNG 文件头读取 `(width, height)`。非 PNG 或失败返回 None。"""
    try:
        with open(path, "rb") as f:
            header = f.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    # IHDR chunk: width(4 bytes) + height(4 bytes) at offset 16-24
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return (width, height)


def _locate_image_output(
    expected_path: str, session_id: Optional[str]
) -> Optional[str]:
    """验证输出文件;不在预期位置就扫 codex 默认目录做 fallback。

    Codex `$imagegen` 默认落盘到 `~/.codex/generated_images/<session>/ig_*.png`,
    有时不会复制到 user 指定路径。这里做一层兜底。
    """
    expected = Path(expected_path)
    if expected.exists() and expected.stat().st_size > 1024:
        return str(expected)

    if not session_id:
        return None

    default_dir = Path.home() / ".codex" / "generated_images" / session_id
    if not default_dir.exists():
        return None

    candidates = sorted(
        default_dir.glob("ig_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    # 只在 fallback 路径才会 import
    import shutil as _shutil  # noqa: PLC0415

    latest = candidates[0]
    expected.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(latest, expected)
    return str(expected)


@mcp.tool(
    name="image_generate",
    description="""
    Dedicated image-generation tool that invokes Codex CLI's `$imagegen` skill
    and uses the Codex subscription to call gpt-image-2.

    Compared to the generic `codex` tool:
      - Silent output: only a structured result is returned (agent reasoning text dropped).
      - Clear parameters: prompt / output_path / aspect_ratio / input_images map 1:1.
      - No automatic retries (avoid double-billing subscription quota).
      - Windows-path safe for input_images.

    Requirements:
      - Codex CLI >= 0.122.0 with `image_generation` feature = stable + true.

    Notes:
      - Codex's raw output is always 1254x1254; it auto-post-processes to the
        target aspect ratio. `original_size` reflects the final file on disk.
      - One logical call may consume >=1 subscription image because Codex
        sometimes self-reviews and regenerates internally.
    """,
    meta={"version": "0.1.0", "author": "dtsgx126"},
)
async def image_generate(
    prompt: Annotated[str, "图像描述,中英文均可"],
    output_path: Annotated[
        str, "最终 PNG 输出路径,相对或绝对均可;函数内部会 resolve"
    ],
    cd: Annotated[Path, "codex exec 的工作目录,必须存在"],
    input_images: Annotated[
        List[Path],
        Field(
            description=(
                "图生图:参考图路径列表;Path 会自动转 str 绕开 Windows Path bug"
            ),
        ),
    ] = [],
    aspect_ratio: Annotated[
        Literal["1:1", "16:9", "9:16", "4:3", "3:4"],
        Field(default="1:1", description="输出比例;Codex 自动后处理裁剪"),
    ] = "1:1",
    timeout_seconds: Annotated[
        int,
        Field(default=300, ge=10, le=900, description="子进程超时(10-900 秒)"),
    ] = 300,
) -> Dict[str, Any]:
    """调用 codex `$imagegen` 出图,静默输出。

    返回:
        {
            "success": bool,
            "path": str,              # 落盘绝对路径
            "session_id": str,        # codex thread_id
            "elapsed_seconds": float,
            "original_size": [w, h] | None,
            "error": str              # 仅 success=False 时有意义
        }
    """
    output_path_abs = str(Path(output_path).resolve())
    cd_str = str(Path(cd).resolve())

    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)

    full_prompt = _build_imagegen_prompt(prompt, output_path_abs, aspect_ratio)
    if os.name == "nt":
        full_prompt = windows_escape(full_prompt)

    cmd = [
        "codex",
        "exec",
        "--cd",
        cd_str,
        "--json",
        "--skip-git-repo-check",
    ]
    if input_images:
        cmd.extend(
            ["--image", ",".join(str(p) for p in input_images)]
        )
    cmd += ["--", full_prompt]

    start = time.monotonic()
    thread_id: Optional[str] = None
    err_message = ""

    try:
        for line in run_shell_command(cmd):
            try:
                line_dict = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if line_dict.get("thread_id"):
                thread_id = line_dict["thread_id"]

            line_type = line_dict.get("type", "")
            if "fail" in line_type:
                err_message += "\n" + line_dict.get("error", {}).get("message", "")
            elif "error" in line_type:
                msg = line_dict.get("message", "")
                if not re.match(r"^Reconnecting\.\.\.\s+\d+/\d+", msg):
                    err_message += "\n" + msg
    except Exception as exc:  # 防御外层子进程异常
        return {
            "success": False,
            "error": f"subprocess 异常: {exc}",
            "path": output_path_abs,
            "session_id": thread_id or "",
            "elapsed_seconds": time.monotonic() - start,
            "original_size": None,
        }

    elapsed = time.monotonic() - start
    actual_path = _locate_image_output(output_path_abs, thread_id)

    if actual_path is None:
        return {
            "success": False,
            "error": (
                "输出文件未生成(目标路径和 fallback 目录都没找到)\n"
                + err_message
            ).strip(),
            "path": output_path_abs,
            "session_id": thread_id or "",
            "elapsed_seconds": elapsed,
            "original_size": None,
        }

    return {
        "success": True,
        "path": actual_path,
        "session_id": thread_id or "",
        "elapsed_seconds": elapsed,
        "original_size": _read_png_size(actual_path),
    }


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
