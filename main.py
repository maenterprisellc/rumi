import yaml
from collectors import  wiki_collector
from preprocess import clean_wiki
from tokenizer.mabpebuilder import MABPECorpusBuilder
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--process" , type=str, help='[clean, collect, buildbpe]')

args = parser.parse_args()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def collect():
    # print("[+] Collecting data from websites...")
    # web_scraper.collect(config["websites"])

    print("[+] Collecting data from Wikipedia...")
    wiki_collector.collect(config["wikipedia"]["topics"], max_pages=20000 , delay=2.5)

    # print("[+] Collecting data from Reddit...")
    # reddit_collector.collect(config["reddit"])

    # print("[+] Collecting data from News API...")
    # news_collector.collect(config["newsapi"]["api_key"])

    # print("[+] Loading local documents...")
    # local_loader.load_from_dir("data/local_docs")

def clean():
    print("[+] Clean data from wikipedia...")
    clean_wiki.process_all()

def main():
    config = load_config()
    if args.process.lower() == 'clean':
        clean()
    elif args.process.lower() == 'collect':
        collect()
    elif args.process.lower() == 'buildbpe':
        builder = MABPECorpusBuilder(clean_text_dir='data/clean/wiki_crawl',
                                     tokenizer_path='data/tokenizer/ma25725.json',
                                     vocab_size=16000)
        builder.train_or_load_tokenizer(force_retrain=False)
    else:
        print("Paramter not supported use wither of these processes: [claen, collect]")


    

if __name__ == "__main__":
    main()
