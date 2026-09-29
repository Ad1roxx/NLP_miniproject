"""Run every phase in order. Cached steps are skipped unless --force is given.

  python run_all.py                     # everything (DistilBERT only if CUDA is available)
  python run_all.py --skip-distilbert   # never train DistilBERT
  python run_all.py --force             # rebuild data, embeddings and models from scratch
"""
import argparse

from src import data, demo_check, eda, error_analysis, evaluate, figures, oos, preprocess, train_sbert, train_tfidf
from src.utils import Timer, cuda_available, log_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-distilbert", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with Timer() as t:
        data.main(args.force)                 # phase 1
        preprocess.save_examples()
        eda.main()                            # phase 2
        train_tfidf.main(args.force)          # phase 3 (M1, M2)
        train_sbert.main(args.force)          # phase 4 (M3)
        if args.skip_distilbert:
            print("DistilBERT skipped (--skip-distilbert)")
        elif not cuda_available():
            print("DistilBERT not run (no GPU)")
            log_run("run_all.py: DistilBERT", 0, "not run (no GPU)")
        else:
            from src import train_distilbert  # imported only when needed (heavier dependencies)
            train_distilbert.main(args.force)  # phase 8 (M4)
        evaluate.main(args.force)             # phase 5: T1, T4, predictions
        oos.main()                            # phase 6: thresholds (validation) then test once
        error_analysis.main()                 # phase 5: T3, T5, error summary
        figures.main()                        # F5-F8
        demo_check.main()                     # demo examples at tuned tau
    log_run("python run_all.py" + (" --skip-distilbert" if args.skip_distilbert else "")
            + (" --force" if args.force else ""), t.seconds, "all phases completed")


if __name__ == "__main__":
    main()
