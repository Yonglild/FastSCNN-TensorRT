import argparse
import struct

import torch
from models.fast_scnn import get_fast_scnn


def main():
    # model = get_fast_scnn(dataset="city", aux=False, nums=19)

    ckpt_path = "/home/wyl/Segmentation/Fast-SCNN-pytorch-master/weights/fast_scnn_citys.pth"
    model = torch.load(ckpt_path, map_location="cpu")

    f = open("./custom.txt", 'w')
    for k, _ in model.items():
        f.write("{}'\n'".format(k))
        print("=> loading {} from pretrained model".format(k))

    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    f = open("fastscnn_custom.wts", "w")
    f.write("{}\n".format(len(model.keys())))
    for k, v in model.items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {} ".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
