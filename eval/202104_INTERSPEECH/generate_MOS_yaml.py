#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import random
import yaml
from pathlib import Path

random.seed(1234)

org_mens = ["SM1", "SM2", "SM3", "SM4"]
org_womens = ["SF1", "SF2", "SF3", "SF4"]
tar_mens = ["TM1", "TM2"]
tar_womens = ["TF1", "TF2"]

with open("MOS_base.yaml") as file:
    yml = yaml.load(file, Loader=yaml.SafeLoader)

ids = []
for i, n in enumerate(range(1, 36)):
    if n < 10:
        n = str(0) + str(n)
    else:
        n = str(n)
    ids.append(n)
vc_dirs = list(Path("./wav").glob("vcc*"))
pwg_dirs = list(Path("./wav").glob("train*"))

method_dict = {
    "wav/vcc2018v1_mlfb512_vqvae_han2han_cs2_train_nodev_all_parallel_wavegan.v1.original": "han2han_cs2",
    "wav/vcc2018v1_mlfb512_vqvae_han2han_noncausal_train_nodev_all_parallel_wavegan.v1.original": "han2han_noncausal",
    "wav/vcc2018v1_mlfb512_vqvae_han2han_cs0_train_nodev_all_parallel_wavegan.v1.original": "han2han_cs0",
    "wav/vcc2018v1_mlfb512_vqvae_sincconv2itu-g_cs0_train_nodev_all_parallel_wavegan.v1.itu-g": "sincconv2itu-g_cs0",
    "wav/vcc2018v1_mlfb512_vqvae_sincconv2itu-g_noncausal_train_nodev_all_parallel_wavegan.v1.itu-g": "sincconv2itu-g_noncausal",
    "wav/vcc2018v1_mlfb512_vqvae_itug2itug_cs0_train_nodev_all_parallel_wavegan.v1.itu-g": "itug2itug_cs0",
    "wav/vcc2018v1_mlfb512_vqvae_itug2itug_noncausal_train_nodev_all_parallel_wavegan.v1.itu-g": "itug2itug_noncausal",
    "wav/vcc2018v1_mlfb512_vqvae_sincconv2itu-g_cs2_train_nodev_all_parallel_wavegan.v1.itu-g": "sincconv2itu-g_cs2",
    "wav/vcc2018v1_mlfb512_vqvae_itug2itug_cs2_train_nodev_all_parallel_wavegan.v1.itu-g": "itug2itug_cs2",
    "wav/train_nodev_all_parallel_wavegan.v1.original": "han_causal_pwg",
    "wav/train_nodev_all_parallel_wavegan.v1.itu-g": "itug_causal_pwg",
    "wav/train_nodev_all_parallel_wavegan.v1.original.noncausal": "han_noncausal_pwg",
}

for j in range(12):
    eval_dict = {}
    p = 0
    for n in range(3):
        for orgs in [org_mens, org_womens]:
            for tars in [tar_mens, tar_womens]:
                org = random.choice(orgs)
                tar = random.choice(tars)
                i = random.choice(ids)
                vc_lbl = "300{}_org-{}_cv-{}.wav".format(i, org, tar)
                pwg_lbl = "{}_300{}_gen.wav".format(tar, i)

                for vc_dir in vc_dirs:
                    m = method_dict[str(vc_dir)]
                    wavf = list(vc_dir.rglob(vc_lbl))
                    eval_dict[f"{m}_{p}"] = str(wavf[0])

                for pwg_dir in pwg_dirs:
                    m = method_dict[str(pwg_dir)]
                    wavf = list(pwg_dir.rglob(pwg_lbl))
                    eval_dict[f"{m}_{p}"] = str(wavf[0])
                p += 1
    yml["pages"][0]["stimuli"] = eval_dict
    yml["testId"] = f"MOS_test{j}"
    # with open(f"./MOS_test{j}.yaml", "w") as fp:
    #     fp.write(yaml.dump(yml, default_flow_style=False))
    with open(f"./config/MOS_yaml/MOS_test{j}.yaml", "w") as fp:
        yaml.dump(yml, fp, encoding="utf8", allow_unicode=True)
