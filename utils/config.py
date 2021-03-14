from xml.dom import minidom
from torch import nn


def parseConfig(xmlFile):
    """Parse the xml configuration defining a structure of a neural network.

    Returns:
    --------
    hp(dict): hyper parameters
    nnParams(dict): neural network parameters
        note that, since Python 3.6, the default dict strcture returned here is an ordered dict.
    """

    document = minidom.parse(xmlFile)
    nn = document.documentElement
    # nodeName vs tagName: node不一定是tag，node包含换行符等
    for node in nn.childNodes:
        if node.nodeName == "hp":
            hpTag = node
        elif node.nodeName == "nnLayer":
            nnLayer = node
    hp = {}
    for node in hpTag.childNodes:
        if type(node) is not minidom.Element:
            continue
        text = node.childNodes[0].data
        try:
            text = float(text)
        except ValueError:
            raise(ValueError("Shouldn't a hyper parameter should always be a digit?"))
        hp[node.tagName] = int(text) if text.is_integer() else text

    nnParams = {}
    for layer in nnLayer.childNodes:
        if type(layer) is not minidom.Element:
            continue
        layerName = layer.tagName + '-' + layer.getAttribute("id")
        if layer.hasAttribute("block"):
            layerName = layer.getAttribute("block") + '-' + layerName
        layerContent = {}
        # layerContent['attribute'] = {key: layer.getAttribute(
        #     key) for key in layer.attributes.keys()}
        layerContent['dim'] = layer.getAttribute('dim')
        for node in layer.childNodes:
            if type(node) is not minidom.Element:
                continue
            text = node.childNodes[0].data
            if text.isdigit():
                layerContent[node.tagName] = int(text)
            elif text == "true":
                layerContent[node.tagName] = True
            elif text == "false":
                layerContent[node.tagName] = False
            else:
                layerContent[node.tagName] = text
            if node.hasAttributes():
                for attri in node.attributes.keys():
                    layerContent[node.tagName+'.' +
                                 attri] = node.getAttribute(attri)
            # if node.tagName == 'activate' and node.hasAttributes():
            #     layerContent[node.tagName] = {'attribute': float(node.getAttribute("param")),
            #                                   'value': layerContent[node.tagName]}
        nnParams[layerName] = layerContent
    return hp, nnParams


def constructOneLayer(layerName, layerContent):
    layerType = layerName.split('-')[-2]
    dim = layerContent['dim']
    assert dim in {"1d", "2d", "3d"}
    if dim == '1d':
        Conv, ConvT = nn.Conv1d, nn.ConvTranspose1d
        BatchNorm = nn.BatchNorm1d
        InstanceNorm = nn.InstanceNorm1d
    if dim == '2d':
        Conv, ConvT = nn.Conv2d, nn.ConvTranspose2d
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    if dim == '3d':
        Conv, ConvT = nn.Conv3d, nn.ConvTranspose3d
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    normalizeType = layerContent['normalize']
    # del layerContent['dim']

    def buildNormLayer(normalizeType, num_features):
        if normalizeType == "null":
            return None
        elif normalizeType == "bn":
            return BatchNorm(num_features)
        elif normalizeType == "in":
            return InstanceNorm(num_features)
        elif normalizeType == "ln":
            shape = layerContent['normalize.dataOutShape']
            shape = tuple(map(int, shape.split(',')[2:]))
            shape = (num_features, *shape)
            return nn.LayerNorm(shape)
        else:
            raise ValueError("Parameter `normalizeType` cannot be resolved!")

    layer = nn.Sequential()
    if layerType == "fc":
        kwargs = {"in_features", "out_features", "bias"}
        kwargs = {key: layerContent[key] for key in kwargs}
        layer.add_module(layerType, nn.Linear(**kwargs))
        if normalizeType != "null":
            layer.add_module(normalizeType,
                             buildNormLayer(normalizeType, kwargs["out_features"]))
    elif layerType == "conv":
        kwargs = {"in_channels", "out_channels",
                  "kernel_size", "stride", "padding", "bias"}
        kwargs = {key: layerContent[key] for key in kwargs}
        layer.add_module(layerType, Conv(**kwargs))
        if normalizeType != "null":
            layer.add_module(normalizeType,
                             buildNormLayer(normalizeType, kwargs["out_channels"]))
    elif layerType == "convT":
        kwargs = {"in_channels", "out_channels", "kernel_size",
                  "stride", "padding", "output_padding", "bias"}
        kwargs = {key: layerContent[key] for key in kwargs}
        layer.add_module(layerType, ConvT(**kwargs))
        if normalizeType != "null":
            layer.add_module(normalizeType,
                             buildNormLayer(normalizeType, kwargs["out_channels"]))
    else:
        raise Exception("xml configuration error!")

    # add the activation fucntion layer
    # activeParam = None
    # if type(layerContent["activate"]) is dict:
    #     activeParam = layerContent["activate"]["attribute"]
    #     layerContent["activate"] = layerContent["activate"]["value"]

    if layerContent["activate"] == "sigmoid":
        layer.add_module("activate", nn.Sigmoid())
    elif layerContent["activate"] == "relu":
        layer.add_module("activate", nn.ReLU())
    elif layerContent["activate"] == "leakyrelu":
        if "activate.param" in layerContent.keys():
            param = float(layerContent["activate.param"])
            layer.add_module("activate", nn.LeakyReLU(param, inplace=True))
        else:
            layer.add_module("activate", nn.LeakyReLU(inplace=True))
    elif layerContent["activate"] == "tanh":
        layer.add_module("activate", nn.Tanh())
    elif layerContent["activate"] == "null":
        pass
    else:
        raise Exception("`activate` is error.")

    return layer


if __name__ == "__main__":
    file = 'particle/nn/config/dcgan.xml'
    hp, nnParams = parseConfig(file)
    print(hp, nnParams, sep='\n')
