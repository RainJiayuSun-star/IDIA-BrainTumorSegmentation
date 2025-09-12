import logging
import random

LOG_LEVEL_EMOJIS = {
    logging.DEBUG: "🐛",
    logging.INFO: "📰",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}

LOG_LEVEL_COLORS = {
    "CRITICAL": "magenta",
    "ERROR": "red",
    "WARNING": "yellow",
    "INFO": "cyan",
    "DEBUG": "green",
}

COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "grey": (128, 128, 128),
}

COLOR_SATURATION_RANGE = (0.4, 0.8)
COLOR_VALUE_RANGE = (0.4, 0.8)

NAME_LENGTHS = {
    "file": 10,
    "function": 10,
    "thread": 5,
    "process": 5,
    "level": 5,
}


def rgb_to_ansi(rgb_color):
    return f"\033[38;2;{rgb_color[0]};{rgb_color[1]};{rgb_color[2]}m"


class CustomFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_colors = {}
        self.process_counter = 0
        self.process_names = {}
        self.thread_colors = {}
        self.level_colors = {
            getattr(logging, level): rgb_to_ansi(COLOR_RGB[color])
            for level, color in LOG_LEVEL_COLORS.items()
        }
        self.thread_counter = 0
        self.thread_names = {}

    def _hsv_to_rgb(self, h, s, v):
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i]

    def _generate_color(self, name, name_colors, name_counter, name_format):
        if name not in name_colors:
            if name.startswith("Main"):
                name_formatted = name
            else:
                if name not in self.process_names:
                    self.process_names[name] = f"{name_format}-{name_counter}"
                    self.process_counter += 1
                name_formatted = self.process_names[name]
            hue = random.random()
            saturation = random.uniform(*COLOR_SATURATION_RANGE)
            value = random.uniform(*COLOR_VALUE_RANGE)
            rgb_color = tuple(
                round(i * 255) for i in self._hsv_to_rgb(hue, saturation, value)
            )
            chosen_color = rgb_to_ansi(rgb_color)
            name_colors[name] = chosen_color
        else:
            name_formatted = self.process_names.get(name, name)
        return name_colors[name], name_formatted

    def format(self, record):
        thread_color, thread_name = self._generate_color(
            record.threadName, self.thread_colors, self.thread_counter, "Thread"
        )
        process_color, process_name = self._generate_color(
            record.processName, self.process_colors, self.process_counter, "Process"
        )

        level_color = self.level_colors.get(
            record.levelno, rgb_to_ansi(COLOR_RGB["grey"])
        )
        emoji = LOG_LEVEL_EMOJIS.get(record.levelno, "")
        message = super().format(record)
        message = message.replace(
            f"{record.asctime} {record.levelname} [{record.module}:{record.funcName}] ",
            "",
        )
        thread_name_formatted = (
            f'{thread_color}[{thread_name:>{NAME_LENGTHS["thread"]}}]'
        )
        process_name_formatted = (
            f'{process_color}[{process_name:>{NAME_LENGTHS["process"]}}]'
        )
        levelname = f'{level_color}[{record.levelname:<{NAME_LENGTHS["level"]}}]'

        file_name_trimmed = (
            (record.module[: NAME_LENGTHS["file"]] + "..py")
            if len(record.module) > NAME_LENGTHS["file"]
            else f"{record.module}.py"
        )
        func_name_trimmed = (
            (record.funcName[: NAME_LENGTHS["function"]] + "..")
            if len(record.funcName) > NAME_LENGTHS["function"]
            else record.funcName
        )
        module_func = f"{level_color}[{file_name_trimmed}:{func_name_trimmed}]"

        message = (
            f"{process_name_formatted} {thread_name_formatted} "
            f"{level_color}[{record.asctime}] {emoji} {levelname} "
            f'{module_func} {message}{rgb_to_ansi(COLOR_RGB["grey"])}'
        )

        return message


def minimize_third_party_logs():
    noisy_logs = ['matplotlib', 'numba', 'PIL', 'plotly']
    [logging.getLogger(name).setLevel(logging.ERROR) for name in noisy_logs]

def setup_logger(log_level: str = "DEBUG"):
    logger = logging.getLogger()

    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if not log_level in log_level_mapping.values():
        log_level = log_level_mapping.get(log_level.upper(), logging.DEBUG)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = CustomFormatter(
        fmt="%(asctime)s %(levelname)s [%(module)s:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
        
    logger.setLevel(log_level)
    minimize_third_party_logs()
