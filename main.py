"""
Single CLI entry point for all smart-home-llm project actions.

Usage:
    python main.py generate          # generate synthetic logs
    python main.py format            # format into training data
    python main.py train             # fine-tune with QLoRA
    python main.py evaluate          # evaluate on val set
    python main.py predict --input <path>   # run inference
    python main.py simulate --log <path>    # run Pygame simulation
    python main.py all               # generate → format → train → evaluate
"""

import sys
import os
import argparse

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)


def _print_stage(name: str):
    print(f"\n{'=' * 60}")
    print(f"=== Stage: {name} ===")
    print(f"{'=' * 60}\n")


def cmd_generate(args):
    _print_stage("generate")
    from data.generator.generate_data import main as gen_main
    gen_args = []
    if getattr(args, "days", None):
        gen_args.extend(["--days", str(args.days)])
    if getattr(args, "output", None):
        gen_args.extend(["--output", args.output])
    if getattr(args, "seed", None):
        gen_args.extend(["--seed", str(args.seed)])
    gen_main(gen_args)


def cmd_format(args):
    _print_stage("format")
    from training.format_data import main as fmt_main
    fmt_main([])


def cmd_train(args):
    _print_stage("train")
    from training.train_lora import train
    train()


def cmd_evaluate(args):
    _print_stage("evaluate")
    from training.evaluate import evaluate
    evaluate()


def cmd_predict(args):
    _print_stage("predict")
    from inference.predict import main as pred_main
    pred_args = []
    if getattr(args, "input", None):
        pred_args.extend(["--input", args.input])
    if getattr(args, "text", None):
        pred_args.extend(["--text", args.text])
    if getattr(args, "max_new_tokens", None):
        pred_args.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if getattr(args, "temperature", None):
        pred_args.extend(["--temperature", str(args.temperature)])
    pred_main(pred_args)


def cmd_simulate(args):
    _print_stage("simulate")
    from simulation.visualize import main as sim_main
    sim_args = []
    if getattr(args, "log", None):
        sim_args.extend(["--log", args.log])
    if getattr(args, "speed", None):
        sim_args.extend(["--speed", str(args.speed)])
    sim_main(sim_args)


def cmd_all(args):
    stages = [
        ("generate", cmd_generate),
        ("format", cmd_format),
        ("train", cmd_train),
        ("evaluate", cmd_evaluate),
    ]

    for name, func in stages:
        try:
            func(args)
        except Exception as e:
            print(f"\n!!! Pipeline FAILED at stage '{name}': {e}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("=== All stages completed successfully! ===")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="smart-home-llm -- CLI entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate                           # generate 90 days of logs
  python main.py format                             # format into train/val
  python main.py train                              # fine-tune Phi-3-mini
  python main.py evaluate                           # evaluate on val set
  python main.py predict --input data/raw/synthetic_logs.jsonl
  python main.py simulate --log data/raw/synthetic_logs.jsonl
  python main.py all                                # full pipeline
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_gen = subparsers.add_parser("generate", help="Generate synthetic smart-home logs")
    p_gen.add_argument("--days", type=int, default=None)
    p_gen.add_argument("--output", type=str, default=None)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_fmt = subparsers.add_parser("format", help="Format raw logs into training data")
    p_fmt.set_defaults(func=cmd_format)

    p_train = subparsers.add_parser("train", help="Fine-tune model with QLoRA")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate fine-tuned model")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = subparsers.add_parser("predict", help="Run inference on logs")
    p_pred.add_argument("--input", type=str, default=None)
    p_pred.add_argument("--text", type=str, default=None)
    p_pred.add_argument("--max-new-tokens", type=int, default=None)
    p_pred.add_argument("--temperature", type=float, default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_sim = subparsers.add_parser("simulate", help="Run Pygame simulation")
    p_sim.add_argument("--log", type=str, required=True)
    p_sim.add_argument("--speed", type=float, default=None)
    p_sim.set_defaults(func=cmd_simulate)

    p_all = subparsers.add_parser("all", help="Run full pipeline: generate -> format -> train -> evaluate")
    p_all.add_argument("--days", type=int, default=None)
    p_all.add_argument("--output", type=str, default=None)
    p_all.add_argument("--seed", type=int, default=None)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
