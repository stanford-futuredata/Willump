from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_model_node import WillumpModelNode

import ast
from willump.willump_utilities import strip_linenos_from_var, weld_scalar_type_to_numpy_type
from weld.types import *


class KerasModelNode(WillumpModelNode, WillumpPythonNode):
    """
    Make predictions using a trees model.
    """
    input_width: int
    model_name: str

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str, model_name: str,
                 input_width, output_type: WeldVec) -> None:
        assert (isinstance(output_type, WeldVec))
        python_ast = self._make_python_ast(input_name, output_name, model_name, output_type)
        super(KerasModelNode, self).__init__(python_ast=python_ast, input_names=[input_name],
                                             output_names=[output_name],
                                             in_nodes=[input_node], output_types=[output_type])
        self.input_width = input_width
        self.model_name = model_name
        self._output_name = output_name
        self._output_type = output_type

    @staticmethod
    def _make_python_ast(input_name: str, output_name: str, model_name: str, output_type: WeldVec) \
            -> ast.AST:
        stripped_output_name = strip_linenos_from_var(output_name)
        stripped_input_name = strip_linenos_from_var(input_name)
        prediction_string = "%s = %s.predict(%s)" % (stripped_output_name,
                                                     model_name, stripped_input_name)
        prediction_ast: ast.Module = ast.parse(prediction_string)
        return prediction_ast.body[0]

    def get_output_name(self):
        return self._output_name

    def get_output_type(self):
        return self._output_type

    def __repr__(self):
        return "Keras Model node with inputs %s\n outputs %s\n and code %s\n" % \
               (str(list(map(lambda x: x.get_output_names(), self._in_nodes))), self._output_names,
                ast.dump(self._python_ast))
