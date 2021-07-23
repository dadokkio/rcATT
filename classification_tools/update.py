import configparser
import argparse

import os
import uuid
import requests
import http.cookiejar
import mimetypes
import pandas as pd
import dask.dataframe as dd

from newspaper import Article, Config
from attackcti import attack_client
from urllib.request import Request, urlopen
from tqdm import tqdm
from tika import parser as pdf_parser
from tqdm.dask import TqdmCallback

import logging

logging.basicConfig(filename="rcatt.log", level=logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

##########################################################
#       LABELS AND DATAFRAME LISTS AND RELATIONSHIP      #
##########################################################

TEXT_FEATURES = ["processed"]
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"


def guess_type_of(link, strict=True):
    link_type, _ = mimetypes.guess_type(link)
    if link_type is None and strict:
        req = Request(link, headers={"User-Agent": "Mozilla/5.0"})
        u = urlopen(req, timeout=10)
        info = u.info()
        link_type = info.get_content_type()  # or using: u.info().gettype()
    return link_type


def parse_report(url):
    try:
        guessed_type = guess_type_of(url)
        cj = http.cookiejar.CookieJar()
        response = requests.get(url, timeout=10, cookies=cj)
        if response.history:
            url = response.url
        if guessed_type in ["text/html", "application/xhtml+xml", "text/plain"]:
            config = Config()
            config.browser_user_agent = USER_AGENT
            article = Article(url, config=config)
            article.download()
            article.parse()
            text = article.text
            logging.debug("[{}] {} OK".format(url, guessed_type))
        elif guessed_type == "application/pdf":
            filename = "/tmp/{}.pdf".format(str(uuid.uuid4()))
            with open(filename, "wb") as f:
                f.write(response.content)
            raw = pdf_parser.from_file(filename)
            text = raw["content"].strip()
            os.remove(filename)
            logging.debug("[{}] {} PDF OK".format(url, filename))
        else:
            logging.error("[{}] {}".format(url, guessed_type))
            return None
        return text.replace("\n", " ")
    except Exception as excp:
        logging.error("[{}] {}".format(url, excp))
        return None


def update_data(output_file=False):
    lift = attack_client()
    config = configparser.ConfigParser()
    all_techniques = lift.get_techniques(stix_format=False)
    all_techniques = lift.remove_revoked(all_techniques)
    all_techniques = lift.remove_deprecated(all_techniques)
    all_tactics = lift.get_tactics()
    all_tactics = lift.remove_revoked(all_tactics)
    all_tactics = lift.remove_deprecated(all_tactics)
    code_tactics = [x["external_references"][0]["external_id"] for x in all_tactics]
    name_tactics = [x["name"] for x in all_tactics]
    slug_name_tactics = [
        x["x_mitre_shortname"].lower().replace(" ", "-") for x in all_tactics
    ]
    code_techniques = []
    name_techniques = []
    stix_ids = []
    relation_df = {}
    urls = {}
    for technique in tqdm(all_techniques):
        technique_id = technique["external_references"][0]["external_id"]
        code_techniques.append(technique_id)
        name_techniques.append(technique["technique"])
        for tactic in technique["tactic"]:
            tactic_id = code_tactics[slug_name_tactics.index(tactic)]
            relation_df.setdefault(tactic_id, [])
            relation_df[tactic_id].append(technique_id)
        stix_ids.append(technique["id"])
        for url in technique["external_references"][1:]:
            if url.get("url", None):
                urls.setdefault(url["url"], {})
                urls[url["url"]][tactic_id] = 1

    if output_file:
        df = pd.DataFrame(urls)
        df = df.transpose()
        df = df.fillna(0)
        df = df.reset_index()
        df.insert(1, "Text", value=None)
        with TqdmCallback(desc="compute"):
            df2 = dd.from_pandas(df, npartitions=20)
            df2["Text"] = df2["index"].apply(parse_report)
            df2 = df2.set_index("index")
            df2.to_csv("data/{}".format(output_file), single_file=True, index=False)

    config["VARIABLES"] = {
        "CODE_TACTICS": code_tactics,
        "NAME_TACTICS": name_tactics,
        "CODE_TECHNIQUES": code_techniques,
        "NAME_TECHNIQUES": name_techniques,
        "STIX_IDENTIFIERS": stix_ids,
        "TACTICS_TECHNIQUES_RELATIONSHIP_D": relation_df,
        "TEXT_FEATURES": ["processed"],
        "ALL_TTPS": code_tactics + code_techniques,
    }
    config["PATH"] = {
        "TRAINING_FILE": "classification_tools/data/{}".format(output_file),
        "ADDED_FILE": "classification_tools/data/training_data_added.csv",
    }
    with open("rcatt.ini", "w") as configfile:
        config.write(configfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch new data from mitre.")
    parser.add_argument("--data", type=str, required=False)
    args = parser.parse_args()
    update_data(args.data)