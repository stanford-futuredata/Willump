from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_model_node import WillumpModelNode

import ast
from willump.willump_utilities import strip_linenos_from_var, weld_scalar_type_to_numpy_type
from weld.types import *


class TreesModelNode(WillumpModelNode, WillumpPythonNode):
    """
    Make predictions using a trees model.
    """
    input_width: int
    model_name: str
    output_type: WeldVec

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str, model_name: str,
                 input_width: int, output_type: WeldVec,
                 predict_proba=False) -> None:
        assert(isinstance(output_type, WeldVec))
        python_ast = self._make_python_ast(input_name, output_name, model_name, predict_proba, output_type)
        super(TreesModelNode, self).__init__(python_ast, [input_name], [output_name], [input_node])
        self.input_width = input_width
        self.model_name = model_name
        self._output_name = output_name
        self.output_type = output_type

    @staticmethod
    def _make_python_ast(input_name: str, output_name: str, model_name: str, predict_proba: bool, output_type: WeldVec)\
            -> ast.AST:
        if predict_proba:
            proba = "_proba"
            proba_selection = "[:,1]"
        else:
            proba = ""
            proba_selection = ""
        stripped_output_name = strip_linenos_from_var(output_name)
        stripped_input_name = strip_linenos_from_var(input_name)
        prediction_string = "%s = %s.predict%s(%s)%s" % (stripped_output_name,
                                                         model_name, proba, stripped_input_name,
                                                         proba_selection)
        prediction_control_string = \
            """if %s.shape[0] > 0:\n""" \
            """\t%s\n""" \
            """else:\n""" \
            """\t%s = scipy.zeros(0, dtype=scipy.%s)\n""" % (stripped_input_name, prediction_string,
                                                             stripped_output_name,
                                                             weld_scalar_type_to_numpy_type(output_type.elemType))
        prediction_ast: ast.Module = ast.parse(prediction_control_string)
        return prediction_ast.body[0]

    def get_output_name(self):
        return self._output_name

    def __repr__(self):
        return "Trees Model node with inputs %s\n outputs %s\n and code %s\n" % \
               (str(list(map(lambda x: x.get_output_names(), self._in_nodes))), self.output_names,
                ast.dump(self._python_ast))
