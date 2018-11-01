import typing


class WillumpGraphNode(object):
    """
    Interface for Willump Graph Nodes.  All types of nodes subclass this class.  Nodes represent
    transforms.
    """
    def get_in_nodes(self) -> typing.List['WillumpGraphNode']:
        """
        Return the node's in-nodes.  These are inputs to the transform represented by the node.
        """
        ...

    def get_node_type(self) -> str:
        """
        Return a string indicating the type of transform the node represents.
        """
        ...

    def get_node_weld(self) -> str:
        """
        Return the Weld IR code defining this node's transform.
        """
        ...
