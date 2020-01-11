import ast

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_model_node import WillumpModelNode
from willump.graph.willump_python_node import WillumpPythonNode

from willump.willump_utilities import *


class WillumpPredictNode(WillumpPythonNode, WillumpModelNode):
    """
    Willump Predict Node.  Contains Python code to predict from a model.
    """

    def __init__(self, model_name: str, x_name: str,
                 x_node: WillumpGraphNode, output_name: str, output_type, input_width: int) -> None:
        self.X_name = x_name
        self.model_name = model_name
        self.X_node = x_node
        self.input_width = input_width
        self._output_name = output_name
        self._output_type = output_type
        predict_statement = "%s = willump_predict_function(%s, %s)" % (strip_linenos_from_var(output_name),
                                                                       model_name, strip_linenos_from_var(x_name))
        predict_ast: ast.Module = ast.parse(predict_statement, "exec").body[0]
        super(WillumpPredictNode, self).__init__(python_ast=predict_ast, input_names=[x_name],
                                                 output_names=[output_name], in_nodes=[x_node],
                                                 output_types=[output_type])

    def get_output_name(self):
        return self._output_name

    def get_output_type(self):
        return self._output_type

    def __repr__(self):
        return "Willump Prediction Python node with inputs %s\n outputs %s\n and code %s\n" % \
               (self.get_in_names(), self.get_output_names(),
                ast.dump(self._python_ast))

    def graphviz_repr(self):
        return "Model:  Make Predictions"
