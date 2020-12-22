class node:
    def __init__(self, name, parent = None):
        self.name = name
        self.parent = parent
        self.children = []
        self.attribute_value = None

    def add_child(self, new_node):
        self.children.append(new_node)

    def delete_child(self, name):
        self.children.remove(x for x in self.children if x.name ==name)


class classifier(node):
    def __init__(self, name, attribute_name, parent = None):
        super().__init__(name, parent)
        self.attribute_name = attribute_name

class final_classifier(node):
    def __init__(self, name, parent = None):
        super().__init__(name, parent)

class road_sign(node):
    def __init__(self, name, properties, parent = None):
        super().__init__(name, parent)
        self.properties = properties



class tree:
    def __init__(self, root):
        self.root = root
        self.nodes = [root]

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, name):
        self.nodes.remove(node for node in self.nodes if node.name == name)