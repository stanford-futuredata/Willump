from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_model_node import WillumpModelNode

from willump.willump_utilities import strip_linenos_from_var
import ast


class WillumpTrainingNode(WillumpPythonNode, WillumpModelNode):
    """
    Willump Training Node.  Contains Python code to train a model.
    """

    def __init__(self, x_name: str, y_name: str, x_node: WillumpGraphNode, y_node: WillumpGraphNode,
                 output_name: str, feature_importances) -> None:
        self.x_name = x_name
        self.y_name = y_name
        self.x_node = x_node
        self.y_node = y_node
        self._output_name = output_name
        train_statement = "%s = willump_train_function(%s, %s)" % (strip_linenos_from_var(output_name),
                                                                   strip_linenos_from_var(x_name),
                                                                   strip_linenos_from_var(y_name))
        train_ast: ast.Module = ast.parse(train_statement, "exec").body[0]
        super(WillumpTrainingNode, self).__init__(python_ast=train_ast, input_names=[x_name, y_name],
                                                  output_names=[output_name], in_nodes=[x_node, y_node],
                                                  output_types=[])
        self._feature_importances = feature_importances
        self.input_width = len(feature_importances)

    def get_feature_importances(self):
        return self._feature_importances

    def get_output_name(self):
        return self._output_name

    def get_output_type(self):
        return None

    def __repr__(self):
        return "Willump Training Python node with inputs %s\n outputs %s\n and code %s\n" % \
               (self.get_in_names(), self.get_output_names(),
                ast.dump(self._python_ast))
