"""DeepResearch Agent 包入口。

这个文件让 `deepresearch` 成为一个 Python 包。它本身不启动 Agent，
真正运行入口在 `__main__.py` 和 `cli.py`。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"


def _print_run_hint() -> None:
    """用户误运行包初始化文件时，提示正确启动方式。"""
    print("deepresearch is a package, not the main runner.")
    print("Use one of these commands from the project root:")
    print("  make run")
    print("  python3 run_agent.py")
    print("  python3 -m deepresearch")


if __name__ == "__main__":
    _print_run_hint()
