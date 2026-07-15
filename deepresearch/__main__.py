"""命令行模块入口。

当你运行 `python3 -m deepresearch` 时，Python 会先进入这个文件，
然后把控制权交给 `cli.main()`。
"""

from .cli import main


if __name__ == "__main__":
    """直接以模块方式运行时启动 CLI。"""
    main()
