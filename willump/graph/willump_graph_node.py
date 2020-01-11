from typing import List

from weld.types import *


class WillumpGraphNode(object):
    """
    Interface for Willump Graph Nodes.  All types of nodes subclass this class.  Nodes represent
    transforms.
    """

    _is_costly_node = False

    def get_in_nodes(self) -> List['WillumpGraphNode']:
        """
        Return the node's in-nodes.  These are inputs to the transform represented by the node.
        """
        ...

    def get_in_names(self) -> List[str]:
        """
        Return the names of a node's input variables.  These come from the in-nodes.
        """
        ...

    def get_output_names(self) -> List[str]:
        """
        Return the names of the variables this node outputs.
        """
        ...

    def get_output_types(self) -> List[WeldType]:
        """
        Return the types of the variables this node outputs.
        """
        ...

    def get_node_weld(self) -> str:
        """
        Return the Weld IR code defining this node's transform.
        """
        ...

    def set_costly_node(self, b: bool):
        self._is_costly_node = b

    def get_costly_node(self) -> bool:
        return self._is_costly_node

    def graphviz_repr(self) -> str:
        return self.__repr__()

