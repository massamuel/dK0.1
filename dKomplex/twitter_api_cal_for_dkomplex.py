#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:05:00 2020

@author: sampoplack
"""


import pandas as pd
import requests
import json
import time
from pandas.io.json import json_normalize
 
df = pd.read_csv('Global_country_populations_2013.csv') 
df = df[['CountryName','lat','lon']]
df.head()