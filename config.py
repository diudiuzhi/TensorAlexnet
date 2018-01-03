# coding=utf-8

import ConfigParser

def get_conf():
    cf = ConfigParser.ConfigParser()
    cf.read("config.ini")
    return cf

