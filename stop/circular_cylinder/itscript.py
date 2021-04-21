#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


def new_f90(n, t):
    fileSource = open("lotus.f90","r")
    fileText = fileSource.read()
    fileSource.close()
    fileLines = fileText.split("\n")
    fileLines[15] = f"  integer         :: ndims = {int(n)}"
    fileLines[31] = f"  real            :: finish = {float(t)}*D"
    fileText = "\n".join(fileLines)
    fileOutput = open("lotus.f90","w")
    fileOutput.write(fileText)
