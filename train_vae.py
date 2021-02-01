#!/usr/bin/python
# coding: utf-8

from wikigen.model.train_edit_vae import main
from wikigen.config import parser, read_config


if __name__ == "__main__":
    args = parser.parse_args()
    args = read_config(args.config)
    print(args)
    main(args)
