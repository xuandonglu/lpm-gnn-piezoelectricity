"""Explicit imag10 adapter; identical indexed message iteration and diagnostics.

The legacy wrapper still accepts only seven imaginary coefficients. This
adapter changes INPUT SCHEMA only and requires the matching complex-loss
graph. No coupling coefficients are dropped or reinterpreted as dielectric.
"""

from .piezo_axisym_complex_loss_messages import AxisymmetricComplexLossMessageGraph
from .piezo_axisym_message_solve import AxisymmetricMessageProblem


class AxisymmetricComplexMessageProblem(AxisymmetricMessageProblem):
    imaginary_parameter_sizes = (7, 10)

    def __init__(self, graph, *args, **kwargs):
        if not isinstance(graph, AxisymmetricComplexLossMessageGraph):
            raise TypeError("The explicit imag10 adapter requires the complex-loss physical graph")
        super().__init__(graph, *args, **kwargs)
        self.metadata.update(
            imaginary_parameter_sizes=list(self.imaginary_parameter_sizes),
            imaginary_schema_adapter_only=True,
            old_imag7_behavior_preserved=True,
        )
