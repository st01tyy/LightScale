import argparse
from light_scale.evaluator import Evaluator, EvaluatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout-only evaluation")
    parser.add_argument("--async_rollout_cfg_path", type=str, required=True)
    parser.add_argument("--rollout_batch_size", type=int, required=True)
    parser.add_argument("--dump_path", type=str, required=True)
    parser.add_argument("--log_file_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--passed_iters", type=int, default=0)
    parser.add_argument("--light_scale_log_level", type=str, default="info")
    parser.add_argument("--init_timeout_seconds", type=float, default=300.0)
    parser.add_argument("--sample_poll_timeout_seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluatorConfig(
        async_rollout_cfg_path=args.async_rollout_cfg_path,
        rollout_batch_size=args.rollout_batch_size,
        dump_path=args.dump_path,
        log_file_path=args.log_file_path,
        n_samples=args.n_samples,
        passed_iters=args.passed_iters,
        light_scale_log_level=args.light_scale_log_level,
        init_timeout_seconds=args.init_timeout_seconds,
        sample_poll_timeout_seconds=args.sample_poll_timeout_seconds,
    )
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()