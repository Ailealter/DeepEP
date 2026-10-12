# Copyright (c) 2026, Lu Lu

"""Extract named kernel definitions for the CPU compilation tests."""

import re


_NON_CODE = re.compile(
    r'//[^\n]*|/\*.*?\*/|R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|^[ \t]*\#(?:[^\n\\]|\\.)*",
    re.DOTALL | re.MULTILINE,
)


def cpp_functions(source, *names):
    """Select unambiguous function definitions, including their template declarations.

    This covers the free functions used by the host kernel tests, without
    preprocessing C++ or resolving overloads. Missing/ambiguous names fail closed.
    Comments and literals retain their offsets but cannot affect brace matching.
    """
    code = _NON_CODE.sub(lambda match: re.sub(r"\S", " ", match.group()), source)

    def closing(start, left, right):
        depth = 0
        for index in range(start, len(code)):
            depth += code[index] == left
            depth -= code[index] == right
            if depth == 0:
                return index + 1
        raise ValueError(f"Unclosed {left!r} in C++ definition")

    definitions = []
    for name in names:
        matches = []
        for match in re.finditer(rf"\b{re.escape(name)}\s*\(", code):
            parameters_end = closing(match.end() - 1, "(", ")")
            body = parameters_end
            while body < len(code) and code[body].isspace():
                body += 1
            if body == len(code) or code[body] != "{":
                continue
            end = closing(body, "{", "}")
            start = max(code.rfind(token, 0, match.start()) for token in ";{}") + 1
            while start < match.start() and code[start].isspace():
                start += 1
            matches.append(source[start:end])
        if len(matches) != 1:
            raise ValueError(f"Expected one C++ definition for {name}, found {len(matches)}")
        definitions.append(matches[0] + "\n")
    return "\n".join(definitions)
