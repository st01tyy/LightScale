import argparse
import dataclasses
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass, field

T = TypeVar("T")

def create_parser_from_dataclass(cls: Type[T], parser: argparse.ArgumentParser) -> None:
    """根据 dataclass 自动向 ArgumentParser 添加参数"""
    for field in dataclasses.fields(cls):
        arg_names = [f"--{field.name}"]
        for alias in field.metadata.get("aliases", []):
            arg_names.append(f"--{alias}")
        arg_type = field.type

        is_optional = not field.metadata.get("required", True)

        kwargs = {"required": not is_optional}

        # 处理 List[T] 和 Optional[List[T]]
        if hasattr(arg_type, "__origin__") and arg_type.__origin__ is list:
            element_type = arg_type.__args__[0]  # 获取 List 内的元素类型
            kwargs["type"] = element_type
            kwargs["nargs"] = "+"  # 允许多个值
        else:
            kwargs["type"] = arg_type

        # 处理默认值
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()

        if arg_type == bool:
            kwargs["action"] = "store_true"
            kwargs.pop("type")

        print(f"{arg_names[0]}, {kwargs}")
        parser.add_argument(*arg_names, dest=field.name, **kwargs)