ESC = "\033"

# Formats don't work consistently accross platforms
FORMATS = {
    "normal": "0",
    "bold": "1",
    # "faint": "2",
    # "underlined": "3",
    # "strikethrough": "4",
}

COLORS = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "gray": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
}


def _generate_ANSI_code(color, format=None):
    format_byte = FORMATS["normal"]
    if format is not None:
        format_byte = FORMATS[format]

    return ESC + "[" + format_byte + ";" + COLORS[color] + "m"


def _reset_ANSI():
    return ESC + "[0m"


def colorize(msg, color, format=None):
    color = color.lower()
    format = format.lower() if format is not None else format

    if color not in COLORS.keys():
        raise Exception(f"Color {color} not recognized.")
    if format is not None and format not in FORMATS.keys():
        raise Exception(f"Format {format} not recognized.")

    return _generate_ANSI_code(color, format) + msg + _reset_ANSI()
