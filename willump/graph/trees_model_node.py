from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_model_node import WillumpModelNode

import ast
from willump.willump_utilities import strip_linenos_from_var


class TreesModelNode(WillumpModelNode, WillumpPythonNode):
    """
    Make predictions using a trees model.
    """
    input_width: int

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str, model_name: str,
                 feature_importances,
                 predict_proba=False) -> None:
        python_ast = self._make_python_ast(input_name, output_name, model_name, predict_proba)
        super(TreesModelNode, self).__init__(python_ast, [input_name], [output_name], [input_node])
        self.input_width = len(feature_importances)

    @staticmethod
    def _make_python_ast(input_name: str, output_name: str, model_name: str, predict_proba: bool) -> ast.AST:
        if predict_proba:
            proba = "_proba"
        else:
            proba = ""
        prediction_string = "%s = %s.predict%s(%s)" % (strip_linenos_from_var(output_name),
                                                       model_name, proba, strip_linenos_from_var(input_name))
        prediction_ast: ast.Module = ast.parse(prediction_string)
        return prediction_ast.body[0]

    def __repr__(self):
        return "Trees Model node with inputs %s\n outputs %s\n and code %s\n" % \
               (str(list(map(lambda x: x.get_output_names(), self._in_nodes))), self.output_names,
                ast.dump(self._python_ast))
