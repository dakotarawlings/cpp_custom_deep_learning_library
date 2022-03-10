# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:13:34 2022

@author: dakot

This script defines a WSGI file that runs our flask application that implements a tool image classification
model on an image file sent via a POST request and returns a class prediction
"""

from app import app as application
if __name__ == "__main__":
    application.run(debug=True)