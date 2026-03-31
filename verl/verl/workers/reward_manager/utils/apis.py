import logging
import json
import re
import random
import requests
import time
import math
import uuid
import pandas as pd
import backoff
import concurrent.futures
import openai
import os
import sys


COVER_VERIFIER_SERVER = os.environ.get('COVER_VERIFIER_SERVER')
COVER_VERIFIER_SERVER_NAME = os.environ.get('COVER_VERIFIER_SERVER_NAME')

if not COVER_VERIFIER_SERVER or not COVER_VERIFIER_SERVER_NAME:
    print("error: COVER_VERIFIER_SERVER not found")
    exit(-1)

CF_VERIFIER_SERVER = os.environ.get('CF_VERIFIER_SERVER')
CF_VERIFIER_SERVER_NAME = os.environ.get('CF_VERIFIER_SERVER_NAME')

if not CF_VERIFIER_SERVER or not CF_VERIFIER_SERVER_NAME:
    print("error: CF_VERIFIER_SERVER not found")
    exit(-1)


def request_cover_check(prompt):
    client = openai.Client(base_url=f"http://{COVER_VERIFIER_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=COVER_VERIFIER_SERVER_NAME,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()

        except:
            time.sleep(1)
            continue
            
    return ""


def request_cf_check(prompt):
    client = openai.Client(base_url=f"http://{CF_VERIFIER_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=CF_VERIFIER_SERVER_NAME,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()

        except:
            time.sleep(1)
            continue
            
    return ""