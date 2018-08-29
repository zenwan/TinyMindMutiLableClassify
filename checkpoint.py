#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'checkpoint management'

import os
import torch

def save_checkpoint(state, address):
    name = 'model_parameters.pth.tar'

    folder = os.path.exists(address)
    if not folder:
        os.mkdir(address)
        print('--- create a new folder ---')
    fulladress = address + '\\' + name
    torch.save(state, fulladress)
    print('model saved:', fulladress)

def load_checkpoint(address):
    name = 'model_parameters.pth.tar'
    fulladress = address + '\\' + name
    return torch.load(fulladress)