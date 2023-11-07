import numpy as np
import torch.nn as nn


def save_file_ply(xyz, rgb, pc_file):
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)
    # print(rgb)
    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
            )
            f.write(str_v)


class attention_manager(object):
    def __init__(self, model, multi_gpu, target_layer="attention_target"):
        self.multi_gpu = multi_gpu
        self.target_layer = target_layer
        self.attention = []
        self.handler = []

        self.model = model

        if multi_gpu:
            self.register_hook(self.model.module)
        else:
            self.register_hook(self.model)

    def register_hook(self, model):
        def get_attention_features(_, inputs, outputs):
            self.attention.append(outputs)

        for name, layer in model._modules.items():
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.register_hook(layer)
            else:
                if name == self.target_layer:
                    handle = layer.register_forward_hook(get_attention_features)
                    self.handler.append(handle)

                else:
                    for name, layer2 in layer._modules.items():
                        if name == self.target_layer:
                            handle = layer2.register_forward_hook(
                                get_attention_features
                            )
                            self.handler.append(handle)

    def remove_hook(self):
        for handler in self.handler:
            handler.remove()
