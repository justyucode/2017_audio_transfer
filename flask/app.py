#!/usr/bin/python
# coding: utf-8
import subprocess
import requests
import flask

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
