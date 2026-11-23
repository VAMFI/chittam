"""Thin entry point kept for ``python eval/benchmark.py``; the implementation lives in
``chittam.benchmark``."""

from chittam.benchmark import (  # noqa: F401
    BenchmarkCase,
    HashingEmbedder,
    _is_leaked_result,
    build_benchmark_cases,
    build_synthetic_corpus,
    load_cases_from_yaml,
    main,
    run_benchmark,
    run_synthetic_benchmark,
)

if __name__ == "__main__":
    main()
