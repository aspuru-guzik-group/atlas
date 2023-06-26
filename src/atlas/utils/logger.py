#!/usr/bin/env python


import sys
import traceback

from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from atlas import __home__


class MessageLogger:

    # DEBUG, INFO           --> stdout
    # WARNING, ERROR, FATAL --> stderr

    VERBOSITY_LEVELS = {
        0: ["FATAL"],
        1: ["FATAL", "ERROR"],
        2: ["FATAL", "ERROR", "WARNING"],
        3: ["FATAL", "ERROR", "WARNING", "STATS"],  # minimal info
        4: ["FATAL", "ERROR", "WARNING", "STATS", "INFO"],  # richer info
        5: ["FATAL", "ERROR", "WARNING", "STATS", "INFO", "DEBUG"],
    }

    WRITER = {
        "DEBUG": sys.stdout,
        "INFO": sys.stdout,
        "WARNING": sys.stderr,
        "ERROR": sys.stderr,
        "FATAL": sys.stderr,
    }

    # more colors and styles:
    # https://stackoverflow.com/questions/2048509/how-to-echo-with-different-colors-in-the-windows-command-line
    # https://joshtronic.com/2013/09/02/how-to-use-colors-in-command-line-output/

    NONE = ""
    WHITE = "#ffffff"
    GREEN = "#d9ed92"
    GRAY = "#d3d3d3"
    YELLOW = "#f9dc5c"
    ORANGE = "#f4a261"
    RED = "#e5383b"
    PURPLE = "#9d4edd"

    COLORS = {
        "DEBUG": GRAY,
        "INFO": NONE,
        "STATS": NONE,
        "WARNING": ORANGE,
        "ERROR": RED,
        "FATAL": PURPLE,
    }

    def __init__(self, name="ATLAS", verbosity=4):
        """
        name : str
            name to give this logger.
        verbosity : int
            verbosity level, between ``0`` and ``4``. with ``0`` only ``FATAL`` messages are shown, with ``1`` also
            ``ERROR``, with ``2`` also ``WARNING``, with ``3`` also ``INFO``, with ``4`` also ``DEBUG``. Default
            is ``3``.
        """
        self.name = name
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]
        self.console = Console(stderr=False)
        self.error_console = Console(stderr=True)

    def update_verbosity(self, verbosity=3):
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]

    def log(self, message, message_type):

        # check if we need to log the message
        if message_type in self.verbosity_levels:
            color = self.COLORS[message_type]
            error_message = None
            if message_type in ["WARNING", "ERROR", "FATAL"]:
                error_message = traceback.format_exc()
                if "NoneType: None" not in error_message:
                    self.error_console.print(error_message, style=f"{color}")

            self.console.print(f"[{message_type}] {message}", style=f"{color}")
            return error_message, message

    def log_chapter(self, title, line="─", style="#34a0a4"):
        if self.verbosity >= 4:
            title = " " + title + " "
            self.console.print(f"{title:{line}^80}", style=style)

    def log_welcome(self, line="─"):
        # if self.verbosity >= 4:
        # image_path = f'{__home__}/ot2/soteria_logo_w_20.png'
        # with Image.open(image_path) as image:
        #     pixels = Pixels.from_image(image)
        #
        # self.console.print(pixels, justify='center')
        self.console.rule()
        msg1 = "\nWelcome to NanoMAP by Soteria Therapeutics!"
        self.console.print(msg1, justify="center", style="#c53a5d bold")
        msg2 = f"Made with :two_hearts: in :Canada:\n"
        self.console.print(msg2, justify="center", style="#e6ad14 bold")
        self.console.rule()

    def log_config(self, full_campaign, campaign_config):

        # -----------------
        # parameter space
        # -----------------
        print("\n")
        table = Table(title="Experiment Configuration Parameter Space")

        table.add_column(
            "Parameter Name", justify="center", style="cyan", no_wrap=True
        )
        table.add_column("Type", justify="center", style="cyan", no_wrap=True)
        table.add_column(
            "Range (# Options)", style="magenta", justify="center"
        )
        table.add_column(
            "Functional?", style="cyan", justify="center", no_wrap=True
        )
        table.add_column(
            "Component Type", style="cyan", justify="center", no_wrap=True
        )
        table.add_column(
            "Target Conc. [mg/mL]",
            style="cyan",
            justify="center",
            no_wrap=True,
        )
        table.add_column(
            "Solvent", style="cyan", justify="center", no_wrap=True
        )

        for param in full_campaign.param_space:
            if param.type == "continuous":
                range_ = f"[{param.low} - {param.high}]"
                num_options = "N/A"
                functional = "No"
            elif param.type == "discrete":
                range_ = f"{min(param.options)} - {max(param.options)}"
                num_options = len(param.options)
                functional = "Yes"
            elif param.type == "categorical":
                range_ = "N/A"
                num_options = len(param.options)
                functional = "Yes"

            table.add_row(
                param.name,
                param.type,
                f"{range_} ({num_options})",
                functional,
                campaign_config["preparation"][param.name]["type"],
                str(campaign_config["preparation"][param.name]["target_conc"]),
                campaign_config["preparation"][param.name]["solvent"],
            )
        console = Console()
        console.print(table)

        # -----------------
        # objective space
        # -----------------
        table = Table(title="Experiment Configuration - Objective Space")

        table.add_column(
            "Objective Name", justify="center", style="cyan", no_wrap=True
        )
        table.add_column("Type", justify="center", style="cyan", no_wrap=True)
        table.add_column("Goal", style="magenta", justify="center")

        for ix, value in enumerate(full_campaign.value_space):
            if len(full_campaign.value_space) == 1:
                goal = full_campaign.goal
            else:
                goal = full_campaign.goal[ix]
            table.add_row(value.name, value.type, goal)

        console = Console()
        console.print(table)
        print("\n")
