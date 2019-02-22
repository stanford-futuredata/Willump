from willump.graph.willump_graph_node import WillumpGraphNode

from typing import Union, Tuple, Mapping


class WillumpModelNode(WillumpGraphNode):
    """
    Willump Model Node.  Represents an arbitrary model, subclassed by all model nodes.

    The model inputs for a model describe the origin of all of a models' feature inputs.  Each model input is a
    tuple of the node that generates the features and a description of what the features' coefficients are in the model.
    This description is either a range of coefficient indices(for vector inputs) or a mapping from column name to
    coefficient index (for dataframe inputs).

    All model inputs must be independent.  None is a dependency of any other.
    """
    _model_inputs = None

    def set_model_inputs(self, model_inputs:
            Mapping[WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]]) -> None:
        self._model_inputs = model_inputs

    def get_model_inputs(self) -> Mapping[WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]]:
        return self._model_inputs

    def get_output_name(self):
        ...

    def get_output_type(self):
        ...
