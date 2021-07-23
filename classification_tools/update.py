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

    config["VARIABLES"] = {
        "CODE_TACTICS": code_tactics,
        "NAME_TACTICS": name_tactics,
        "CODE_TECHNIQUES": code_techniques,
        "NAME_TECHNIQUES": name_techniques,
        "STIX_IDENTIFIERS": stix_ids,
        "TACTICS_TECHNIQUES_RELATIONSHIP_D": relation_df,
        "TEXT_FEATURES": ["processed"],
        "TRAINING_FILE": "data/{}".format(output_file),
        "ALL_TTPS": [
            "TA0006",
            "TA0002",
            "TA0040",
            "TA0003",
            "TA0004",
            "TA0008",
            "TA0005",
            "TA0010",
            "TA0007",
            "TA0009",
            "TA0011",
            "TA0001",
            "T1066",
            "T1047",
            "T1156",
            "T1113",
            "T1067",
            "T1037",
            "T1033",
            "T1003",
            "T1129",
            "T1492",
            "T1044",
            "T1171",
            "T1014",
            "T1501",
            "T1123",
            "T1133",
            "T1109",
            "T1099",
            "T1069",
            "T1114",
            "T1163",
            "T1025",
            "T1116",
            "T1093",
            "T1178",
            "T1013",
            "T1192",
            "T1489",
            "T1206",
            "T1063",
            "T1080",
            "T1167",
            "T1165",
            "T1137",
            "T1089",
            "T1487",
            "T1214",
            "T1119",
            "T1115",
            "T1103",
            "T1007",
            "T1040",
            "T1135",
            "T1120",
            "T1082",
            "T1071",
            "T1053",
            "T1162",
            "T1176",
            "T1106",
            "T1058",
            "T1202",
            "T1024",
            "T1091",
            "T1005",
            "T1140",
            "T1195",
            "T1190",
            "T1219",
            "T1079",
            "T1036",
            "T1055",
            "T1205",
            "T1218",
            "T1038",
            "T1050",
            "T1010",
            "T1032",
            "T1062",
            "T1182",
            "T1029",
            "T1004",
            "T1009",
            "T1076",
            "T1131",
            "T1181",
            "T1483",
            "T1185",
            "T1021",
            "T1207",
            "T1107",
            "T1145",
            "T1112",
            "T1491",
            "T1155",
            "T1217",
            "T1183",
            "T1085",
            "T1031",
            "T1092",
            "T1222",
            "T1179",
            "T1019",
            "T1042",
            "T1117",
            "T1054",
            "T1108",
            "T1193",
            "T1215",
            "T1101",
            "T1177",
            "T1125",
            "T1144",
            "T1045",
            "T1016",
            "T1198",
            "T1087",
            "T1090",
            "T1059",
            "T1482",
            "T1175",
            "T1020",
            "T1070",
            "T1083",
            "T1138",
            "T1191",
            "T1188",
            "T1074",
            "T1049",
            "T1064",
            "T1051",
            "T1497",
            "T1102",
            "T1104",
            "T1480",
            "T1204",
            "T1196",
            "T1057",
            "T1141",
            "T1041",
            "T1060",
            "T1023",
            "T1026",
            "T1122",
            "T1015",
            "T1212",
            "T1210",
            "T1142",
            "T1199",
            "T1098",
            "T1170",
            "T1048",
            "T1097",
            "T1110",
            "T1001",
            "T1039",
            "T1078",
            "T1073",
            "T1068",
            "T1208",
            "T1027",
            "T1201",
            "T1187",
            "T1486",
            "T1488",
            "T1174",
            "T1002",
            "T1081",
            "T1128",
            "T1056",
            "T1203",
            "T1168",
            "T1100",
            "T1186",
            "T1184",
            "T1095",
            "T1075",
            "T1012",
            "T1030",
            "T1028",
            "T1034",
            "T1499",
            "T1065",
            "T1197",
            "T1088",
            "T1493",
            "T1132",
            "T1500",
            "T1223",
            "T1213",
            "T1194",
            "T1200",
            "T1485",
            "T1130",
            "T1022",
            "T1189",
            "T1498",
            "T1158",
            "T1221",
            "T1134",
            "T1209",
            "T1111",
            "T1159",
            "T1136",
            "T1018",
            "T1046",
            "T1052",
            "T1105",
            "T1084",
            "T1160",
            "T1484",
            "T1220",
            "T1173",
            "T1008",
            "T1096",
            "T1124",
            "T1035",
            "T1086",
            "T1490",
            "T1216",
            "T1094",
            "T1043",
            "T1211",
            "T1127",
            "T1077",
        ],
    }
    with open("rcatt.ini", "w") as configfile:
        config.write(configfile)

    if output_file:
        df = pd.DataFrame(urls)
        df = df.transpose()
        df = df.fillna(0)
        df = df.reset_index()
        with TqdmCallback(desc="compute"):
            df2 = dd.from_pandas(df, npartitions=20)
            df2["text"] = df2["index"].apply(parse_report)
            df2.to_csv("data/{}".format(output_file), single_file=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch new data from mitre.")
    parser.add_argument("--data", type=str, required=False)
    args = parser.parse_args()
    update_data(args.data)
