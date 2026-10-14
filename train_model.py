import argparse, json
from detector.config import ROOT
from detector.training import train_all
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate news-text classifiers.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "news_dataset.csv"))
    parser.add_argument("--output", default=str(ROOT / "artifacts"))
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_all(args.dataset, args.output, tune=not args.no_tune), indent=2))
if __name__ == "__main__":
    main()
