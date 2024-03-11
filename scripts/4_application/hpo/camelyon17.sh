#!/bin/bash

python src/main.py hparams_search=ax_camelyon17 local=encephalon 
python src/main.py hparams_search=ax_camelyon17 local=encephalon train/network=eq_nasnet

