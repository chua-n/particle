from xml.dom import minidom
from torch import nn


def parseConfig(xmlFile):
    """Parse the xml configuration defining a structure of a neural network.

    Returns:
    --------
    nnParams(dict): note that, since Python 3.6, the default dict strcture 
        returned here is an ordered dict.
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
        kwargs = {}
        for param in layer.childNodes:
            if type(param) is not minidom.Element:
                continue
            text = param.childNodes[0].data
            if text.isdigit():
                kwargs[param.tagName] = int(text)
            elif text == "true":
                kwargs[param.tagName] = True
            elif text == "false":
                kwargs[param.tagName] = False
            else:
                kwargs[param.tagName] = text
            if param.tagName == 'activate_mode' and param.hasAttributes():
                kwargs[param.tagName] = {'attribute': float(param.getAttribute("param")),
                                         'value': kwargs[param.tagName]}
        nnParams[layer.tagName+"_"+layer.getAttribute("id")] = kwargs
    return hp, nnParams


def constructOneLayer(layerType, layerParam):
    layer = nn.Sequential()
    if layerType.startswith("fc"):
        kwargs = {"in_features", "out_features", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, nn.Linear(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", nn.BatchNorm1d(kwargs["out_features"]))
    elif layerType.startswith("convTranspose"):
        ConvTranspose, BatchNorm = (nn.ConvTranspose2d, nn.BatchNorm2d) if "2d" in layerType \
            else (nn.ConvTranspose3d, nn.BatchNorm3d)
        kwargs = {"in_channels", "out_channels", "kernel_size",
                  "stride", "padding", "output_padding", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, ConvTranspose(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", BatchNorm(kwargs["out_channels"]))
    elif layerType.startswith("conv"):
        Conv, BatchNorm = (nn.Conv2d, nn.BatchNorm2d) if "2d" in layerType \
            else (nn.Conv3d, nn.BatchNorm3d)
        kwargs = {"in_channels", "out_channels",
                  "kernel_size", "stride", "padding", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, Conv(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", BatchNorm(kwargs["out_channels"]))
    else:
        raise Exception("xml configuration error!")

    # add the activation fucntion layer
    activeParam = None
    if type(layerParam["activate_mode"]) is dict:
        activeParam = layerParam["activate_mode"]["attribute"]
        layerParam["activate_mode"] = layerParam["activate_mode"]["value"]

    if layerParam["activate_mode"] == "sigmoid":
        layer.add_module("activate", nn.Sigmoid())
    elif layerParam["activate_mode"] == "relu":
        layer.add_module("activate", nn.ReLU())
    elif layerParam["activate_mode"] == "leakyrelu":
        if activeParam is not None:
            layer.add_module("activate", nn.LeakyReLU(activeParam))
        else:
            layer.add_module("activate", nn.LeakyReLU())
    elif layerParam["activate_mode"] == "tanh":
        layer.add_module("activate", nn.Tanh())
    elif layerParam["activate_mode"] == "null":
        pass
    else:
        raise Exception("`activate_mode` is error.")

    return layer


if __name__ == "__main__":
    file = 'particle/nn/config/dcgan.xml'
    hp, nnParams = parseConfig(file)
    print(hp, nnParams)
