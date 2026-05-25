from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PlotFontSizes:
    """Shared typography settings for matplotlib probe result plots.

    Individual plots can use ``scaled`` or ``with_overrides`` to keep a local
    override while still inheriting the project-wide defaults.
    """

    axis_title: float = 18
    axis_label: float = 16
    tick_label: float = 14
    legend: float = 16
    figure_title: float = 20
    combined_figure_title: float = 24
    annotation: float = 9

    def scaled(self, factor: float, **overrides: float) -> "PlotFontSizes":
        scaled_sizes = replace(
            self,
            axis_title=self.axis_title * factor,
            axis_label=self.axis_label * factor,
            tick_label=self.tick_label * factor,
            legend=self.legend * factor,
            figure_title=self.figure_title * factor,
            combined_figure_title=self.combined_figure_title * factor,
            annotation=self.annotation * factor,
        )
        return replace(scaled_sizes, **overrides)

    def with_overrides(self, **overrides: float) -> "PlotFontSizes":
        return replace(self, **overrides)


PROBE_PLOT_FONT_SIZES = PlotFontSizes()
