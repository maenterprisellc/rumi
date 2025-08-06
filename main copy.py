import yaml
from collectors import  wiki_collector
from preprocess import clean_wiki
from tokenizer.mabpebuilder import MABPECorpusBuilder
import argparse
import logging
import signal
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--process" , type=str, help='[clean, collect, buildbpe]')

args = parser.parse_args()

# Setup logging to file for service use
logger = logging.getLogger("rumi_service")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("rumi_service.log", mode="a", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

shutdown_flag = False

def handle_shutdown(signum, frame):
    global shutdown_flag
    logger.info(f"Received shutdown signal: {signum}")
    shutdown_flag = True

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def collect(config):
    logger.info("Collecting data from Wikipedia...")
    wiki_collector.collect(config["wikipedia"]["topics"], max_pages=20000 , delay=2.5)


def clean(config):
    logger.info("Clean data from wikipedia...")
    clean_wiki.process_all()


def main():
    global shutdown_flag
    config = load_config()
    logger.info(f"Service started with process: {args.process}")
    if args.process.lower() == 'clean':
        clean(config)
    elif args.process.lower() == 'collect':
        collect(config)
    elif args.process.lower() == 'buildbpe':
        builder = MABPECorpusBuilder(clean_text_dir=config["train_tokenizer"]["data_path"],
                                     tokenizer_path=config["train_tokenizer"]["tokenizer_path"],
                                     vocab_size=config["train_tokenizer"]["vocab_size"])
        builder.train_or_load_tokenizer(force_retrain=False)
    else:
        logger.error("Parameter not supported. Use one of: [clean, collect, buildbpe]")
    # Service loop (wait for shutdown)
    while not shutdown_flag:
        signal.pause()
    logger.info("Service shutting down.")


if __name__ == "__main__":
    main()
