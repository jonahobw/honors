import torch

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

    def predict(self, prev_edge_weight, image, prediction_vector=None, gpu_id = None):
        # previous edge weight is the probability of getting to the current node, the node's prediction vector
        # will be multiplied by this
        # function will return a dict of the format {<class name> : <probability that input image is <class name>>}

        if prediction_vector is None:
            prediction_vector = {}

        # if this is a none node, give it no probability
        if(self.name.find("none")>0):
            return prediction_vector

        # base case, if this is a road sign, return the previous edge weight
        if(isinstance(self, road_sign)):
            prediction_vector[self.name] = prev_edge_weight
            return prediction_vector

        if(isinstance(self, classifier) or isinstance(self, final_classifier)):
            prediction = get_model_prediction_probs(self.neuralnet, image)

            # map prediction output to child nodes and recursively call predict()
            for i, child in enumerate(self.pred_value_names):
                next_prev_edge_weight = prev_edge_weight * prediction[i]
                child.predict(next_prev_edge_weight, image, prediction_vector, gpu_id = gpu_id)

        return prediction_vector




class classifier(node):
    def __init__(self, name, attribute_name, parent = None):
        super().__init__(name, parent)
        self.attribute_name = attribute_name
        self.neuralnet = None
        # maps the output of the neural network to the children of current node, the elements of the array should
        # be the child nodes and they should be in the order of the neural network output vector
        self.pred_value_names = []

class final_classifier(node):
    def __init__(self, name, parent = None):
        super().__init__(name, parent)
        self.neuralnet = None
        # maps the output of the neural network to the children of current node, the elements of the array should
        # be the child nodes and they should be in the order of the neural network output vector
        self.pred_value_names = []

class road_sign(node):
    def __init__(self, name, properties, parent = None):
        super().__init__(name, parent)
        self.properties = properties



class tree():
    def __init__(self, root):
        self.root = root
        self.nodes = [root]

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, name):
        self.nodes.remove(node for node in self.nodes if node.name == name)


# functions from pytorch_resnet.py repeated here to avoid circular import
def get_model_prediction_probs(model, image, gpu_id = None):
    # feeds an image to a neural network and returns the predictions vector
    model.eval()
    image = image.clone().detach().unsqueeze(0)
    if torch.cuda.is_available():
        image= image.to('cuda')
        model.to('cuda')
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            model.cuda(gpu_id)
            image = image.to(gpu_id)
        else:
            image = image.to('cuda')
            model.to('cuda')

    with torch.no_grad():
        output = model(image)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    sm = torch.nn.functional.softmax(output[0], dim=0)
    sm_list = sm.tolist()

    return sm_list