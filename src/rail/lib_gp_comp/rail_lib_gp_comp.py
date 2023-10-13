#! /usr/bin/env python

# Copyright (C) 2022 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules

# creation imports


def check_implemented_components(input_component):
    """

    Parameters
    ----------
    input_component

    Returns
    -------

    """
    implemented_modelling_components = ['Schechter']

    if input_component in implemented_modelling_components:
        return True
    else:
        return False
