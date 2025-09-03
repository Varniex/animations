from __future__ import annotations
from manimlib.constants import OUT
from custom.constants import CYAN
from manimlib.animation.specialized import LaggedStart
from manimlib.animation.growing import GrowFromPoint

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3Array
    from manimlib.mobject.mobject import Mobject


class Wavy(LaggedStart):
    def __init__(
        self,
        mobject: Mobject,
        point: Vect3Array = None,
        color: ManimColor = CYAN,
        **kwargs,
    ):
        point = OUT if point is None else point
        super().__init__(
            *[
                GrowFromPoint(m, point=point, point_color=color)
                for m in mobject.family_members_with_points()
            ],
            **kwargs,
        )
