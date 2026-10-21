"""Validate input, run selected backends, print results and log the invocation."""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List

import sympy as sp

from .limits import (
    SUPPORTED_N_POINTS,
    DEFAULT_MAX_STEPS,
    DEFAULT_PER_STEP_TIMEOUT,
    DEFAULT_SAMPLE_TIMEOUT,
    DEFAULT_SFT_BEAM_SIZE,
    DEFAULT_SFT_BEAM_TEMPERATURE,
    DEFAULT_CDS_BEAM_SIZE,
    DEFAULT_CDS_TEMPERATURE,
    DEFAULT_CDS_NUCLEUS_P,
    DEFAULT_CDS_LENGTH_PENALTY,
    DEFAULT_SIMPLIFY_MODE,
    SIMPLIFY_MODES,
    DEFAULT_SFT_PATIENCE,
    DEFAULT_GBFS_MAX_APPS,
    DEFAULT_GBFS_MAX_SECONDS,
    DEFAULT_GBFS_MAX_TERMS,
)
from .validate import validate, _get_bracket_classes, SIZE_LIMIT_TAGS, ENGINE_LIMIT_TAGS
from .limits import (DEFAULT_GBFS_ORDERING, DEFAULT_GBFS_TARGET_CRITERION, DEFAULT_GBFS_THREADS,
                     DEFAULT_GBFS_SPLIT_MAX_TERMS, DEFAULT_GBFS_C_CANCEL_MAX_TERMS,
                     DEFAULT_POST_TIMEOUT_S, DEFAULT_ALGO_GRACE_S, DEFAULT_MAX_PARALLEL_ALGOS,
                     GBFS_ORDERINGS, GBFS_TARGET_CRITERIA)
from .atomize import atomize, AtomizeError, has_opaque_den
from .parse import parse_expr
from .runner import run_all, list_algorithms, is_gbfs, is_known
from .log import utc_now_iso, new_invocation_id, append_invocation, resolve_log_path
from .result import CLIInvocation


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ampzip",
        description="Simplify massless four-, five- and six-point spinor-helicity expressions. "
                    "See docs/LIMITS.md for supported inputs and parameter limits.",
        epilog="SFT and CDS require separate dependencies and model checkpoints. "
               "See ampzip fetch-backends --help.",
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--expr", type=str,
                     help="Expression to simplify, using ab(), sb(), s() or chain() notation.")
    inp.add_argument("--expr-file", type=str,
                     help="Read the expression from the first line of a file.")

    p.add_argument("--n-point", type=str, choices=["auto", "4", "5", "6"], default="auto",
                   help="Number of external legs. auto uses the largest leg label in the expression.")
    p.add_argument("--target", type=str, default=None,
                   help="Target expression for success evaluation. A result is solved if it passes "
                        "the finite-field equality check against the target and meets "
                        "--gbfs-target-criterion.")

    p.add_argument("--algorithms", type=str, default="model_free_gbfs",
                   help=f"Comma-separated backends to run. Available: {', '.join(list_algorithms())}.")

    # --- learned backends
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help="Maximum number of steps per SFT rollout.")
    p.add_argument("--per-step-timeout", type=float, default=DEFAULT_PER_STEP_TIMEOUT,
                   help="Time limit in seconds for one SFT identity application in SymPy. "
                        "The step is abandoned if it exceeds the limit.")
    p.add_argument("--sample-timeout", type=float, default=DEFAULT_SAMPLE_TIMEOUT,
                   help="Time limit in seconds for one SFT rollout, checked between steps. "
                        "For CDS, limits the wait for decoding, excluding checkpoint loading and "
                        "post-processing. On timeout, CDS reports no expression. Decoding continues, "
                        "and the CLI waits for it to finish before exiting. "
                        "Use --gbfs-max-seconds for GBFS.")
    p.add_argument("--sft-beam-size", type=int, default=DEFAULT_SFT_BEAM_SIZE,
                   help="Number of rollouts for sft_best_of_n: one greedy rollout and N-1 sampled rollouts.")
    p.add_argument("--sft-beam-temperature", type=float, default=DEFAULT_SFT_BEAM_TEMPERATURE,
                   help="sft_best_of_n: softmax temperature of the sampled rollouts.")
    p.add_argument("--sft-patience", type=int, default=DEFAULT_SFT_PATIENCE,
                   help="SFT: stop after N consecutive steps without improving the bracket count "
                        "or, at equal bracket count, the term count. 0 disables this stopping rule.")
    p.add_argument("--cds-beam-size", type=int, default=DEFAULT_CDS_BEAM_SIZE,
                   help="CDS: beam width.")
    p.add_argument("--cds-temperature", type=float, default=DEFAULT_CDS_TEMPERATURE,
                   help="CDS: temperature for nucleus sampling.")
    p.add_argument("--cds-nucleus-p", type=float, default=DEFAULT_CDS_NUCLEUS_P,
                   help="CDS: probability threshold for nucleus sampling. 0 disables nucleus sampling.")
    p.add_argument("--cds-length-penalty", type=float, default=DEFAULT_CDS_LENGTH_PENALTY,
                   help="CDS: beam length penalty. Higher values favour longer outputs.")

    # --- search family (model_free_gbfs, gbfs_c, gbfs_d, gbfs_e, gbfs_f)
    p.add_argument("--gbfs-max-apps", type=int, default=DEFAULT_GBFS_MAX_APPS,
                   help="Move budget for each GBFS variant.")
    p.add_argument("--gbfs-max-seconds", type=float, default=DEFAULT_GBFS_MAX_SECONDS,
                   help="Search time limit in seconds for each GBFS variant. Returns the best "
                        "expression found. An operation already in progress is allowed to finish.")
    p.add_argument("--gbfs-max-terms", type=int, default=DEFAULT_GBFS_MAX_TERMS,
                   help="Maximum numerator term count during input expansion and identity application. Moves exceeding the limit are skipped. Inputs exceeding it are rejected.")
    p.add_argument("--gbfs-ordering", choices=list(GBFS_ORDERINGS), default=DEFAULT_GBFS_ORDERING,
                   help="Frontier ordering. factored uses the variant's default ranking. "
                        "raw uses the raw numerator bracket count. Does not apply to gbfs_c.")
    p.add_argument("--gbfs-den-aware", action="store_true",
                   help="Use denominator factors to adjust frontier ordering. Cannot be combined "
                        "with --gbfs-ordering raw. Does not apply to gbfs_c.")
    p.add_argument("--gbfs-target-criterion", choices=list(GBFS_TARGET_CRITERIA),
                   default=DEFAULT_GBFS_TARGET_CRITERION,
                   help="Success criterion with --target. brackets requires a raw numerator bracket "
                        "count no greater than the target's. brackets-and-terms also requires a "
                        "numerator term count no greater than the target's. none checks equality only.")
    p.add_argument("--gbfs-threads", type=int, default=DEFAULT_GBFS_THREADS,
                   help="Worker threads for move application. Parallel move application preserves "
                        "the search order. The returned expression may differ if --gbfs-max-seconds "
                        "is reached or the search is cancelled. 0 selects the thread count automatically.")
    p.add_argument("--gbfs-audit-steps", action="store_true",
                   help="Check each step of the returned derivation over finite fields. "
                        "Applies to model_free_gbfs and gbfs_d.")
    p.add_argument("--gbfs-split-max-terms", type=int, default=DEFAULT_GBFS_SPLIT_MAX_TERMS,
                   help="Maximum number of terms searched separately by gbfs_e and gbfs_f. "
                        "If splitting the original input exceeds the limit, numerator splitting "
                        "is attempted before falling back to a full-expression search. "
                        "0 disables the limit.")
    p.add_argument("--gbfs-c-cancel-max-terms", type=int, default=DEFAULT_GBFS_C_CANCEL_MAX_TERMS,
                   help="gbfs_c: maximum numerator or denominator term count before SymPy "
                        "cancellation. Expressions exceeding the limit are skipped. 0 disables the limit.")

    # --- output, bounds and logging
    p.add_argument("--simplify-mode", choices=list(SIMPLIFY_MODES),
                   default=DEFAULT_SIMPLIFY_MODE,
                   help="SymPy post-processing of returned expressions. none leaves them unchanged. "
                        "cancel cancels common polynomial factors. expand cancels factors and then "
                        "expands the expression. simplify applies SymPy's simplify function.")
    p.add_argument("--post-process-timeout", type=float, default=DEFAULT_POST_TIMEOUT_S,
                   help="Time limit in seconds for each post-processing operation. On timeout, "
                        "returns the backend output without post-processing. 0 disables the limit.")
    p.add_argument("--max-parallel-algos", type=int, default=DEFAULT_MAX_PARALLEL_ALGOS,
                   help="Maximum number of backends running concurrently. 1 runs them sequentially.")
    p.add_argument("--algo-grace-seconds", type=float, default=DEFAULT_ALGO_GRACE_S,
                   help="Additional time in seconds to wait for a backend after its time budget "
                        "expires when backends run in parallel.")
    p.add_argument("--capture-derivation", action="store_true",
                   help="Include available derivation steps and intermediate expressions in the log.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Directory for cli.jsonl. Defaults to AMPZIP_LOG_DIR or the user log directory.")
    p.add_argument("--json", action="store_true",
                   help="Also print the full result record as JSON on stdout.")
    return p


def _read_expr_arg(args) -> str:
    if args.expr is not None:
        return args.expr
    with open(args.expr_file, "r", encoding="utf-8") as f:
        return f.readline().strip()


def _print_validation(v) -> None:
    if v.ok:
        print(f"[validate] OK  (n_point={v.inferred_n_point})")
        print(f"[validate] canonical: {v.canonical_expr_str}")
        return
    print("[validate] FAILED — input rejected:")
    for iss in v.issues:
        print(f"  - {iss.tag}: {iss.message}")
    if v.canonical_expr_str:
        print(f"[validate] parsed expression: {v.canonical_expr_str}")


def _print_results(results) -> None:
    """Print factored numerator counts for each backend.

    `pre_final` and `pre_min` are counts before post-processing. `rawnum` is the
    raw numerator bracket count of the selected expression when available.
    """
    show_raw = any(
        (r.extra.get("simplify_mode") or "none") != "none"
        and (
            r.extra.get("raw_bracket_count_final", r.bracket_count_final) != r.bracket_count_final
            or r.extra.get("raw_bracket_count_min", r.bracket_count_min) != r.bracket_count_min
        )
        for r in results
    )
    print()
    head = f"{'Algorithm':<16} {'solved':<7} {'start':<5} "
    head += f"{'pre_final':<9} {'final':<5} {'pre_min':<7} {'min':<5} " if show_raw else f"{'final':<5} {'min':<5} "
    head += f"{'rawnum':<6} {'steps':<5} {'sec':<7} {'stop':<14}"
    print(head)
    print("-" * len(head))
    for r in results:
        ex = r.extra or {}
        rawnum = ex.get("raw_numerator_brackets_min")
        rawnum = "-" if rawnum is None else rawnum
        row = f"{r.algorithm:<16} {str(r.solved):<7} {r.bracket_count_start:<5} "
        if show_raw:
            row += (f"{ex.get('raw_bracket_count_final', r.bracket_count_final):<9} "
                    f"{r.bracket_count_final:<5} "
                    f"{ex.get('raw_bracket_count_min', r.bracket_count_min):<7} "
                    f"{r.bracket_count_min:<5} ")
        else:
            row += f"{r.bracket_count_final:<5} {r.bracket_count_min:<5} "
        row += f"{rawnum:<6} {r.n_steps:<5} {r.duration_s:<7.2f} {str(ex.get('stop_reason', '-')):<14}"
        print(row)
    for r in results:
        ex = r.extra or {}
        if ex.get("decode_still_running"):
            print(f"  {r.algorithm}: Timed out. Decoding continues. No expression was returned.")
        if r.min_expr_str and r.min_expr_str != r.final_expr_str:
            print(f"  final_expr = {r.final_expr_str}")
            print(f"  min_expr   = {r.min_expr_str}")
        else:
            print(f"  final_expr = {r.final_expr_str}")
        rec = ex.get("reconstructed_min") or ex.get("reconstructed_final")
        if rec and rec not in (r.min_expr_str, r.final_expr_str):
            print(f"  reconstructed = {rec}")
        skips = {k: v for k, v in ex.items() if k.endswith("_skips") and v}
        if skips:
            print(f"  skipped moves: " + ", ".join(f"{k}={v}" for k, v in skips.items()))
        if ex.get("post_process_timed_out"):
            print("  Post-processing exceeded --post-process-timeout. "
                  "Showing the backend output without post-processing.")
        if ex.get("overran"):
            print("  Result received after the time budget plus --algo-grace-seconds.")
        if ex.get("error"):
            print(f"  ERROR: {ex['error']}")


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "fetch-backends":
        from .fetch_backends import main as fetch_main
        return fetch_main(argv[1:])
    t_main = time.time()
    args = _build_argparser().parse_args(argv)
    if args.gbfs_max_terms < 1:
        print("[error] --gbfs-max-terms must be at least 1")
        return 2
    if args.gbfs_den_aware and args.gbfs_ordering == "raw":
        print("[error] --gbfs-den-aware cannot be combined with --gbfs-ordering raw")
        return 2
    n_override = None if args.n_point == "auto" else int(args.n_point)
    raw_input = _read_expr_arg(args)
    inv_id = new_invocation_id()
    ts = utc_now_iso()

    record = CLIInvocation(
        timestamp_utc=ts,
        invocation_id=inv_id,
        cwd=os.getcwd(),
        cli_args=vars(args),
        raw_input=raw_input,
        validation=None,  # filled below
    )

    # 0. Atomize: rewrite s()/chain() notation into pure ab/sb before anything else.
    #    The original raw_input is kept in `record`; the atomized form feeds the pipeline.
    try:
        raw_input = atomize(raw_input, n_hint=n_override)
        atomized_target = atomize(args.target, n_hint=n_override) if args.target else None
    except AtomizeError as e:
        print(f"[atomize] FAILED — input rejected:\n  - {e}")
        record.error = f"atomize_failed: {e}"
        record.total_duration_s = time.time() - t_main
        log_path = append_invocation(record, log_dir=args.log_dir)
        print(f"\n[log] {log_path}  ({record.total_duration_s:.2f}s)", file=sys.stderr)
        return 2

    # 1. Validate. enforce_size_limits=False so the SFT/CDS size budget never rejects the
    #    whole run — a too-big expression is still fair game for the model-free gbfs family.
    #    Size violations are reported in v.issues and handled by the size route in step 3.
    t_v = time.time()
    v = validate(raw_input, override_n_point=n_override, enforce_size_limits=False)
    record.validation_duration_s = time.time() - t_v
    record.validation = v
    _print_validation(v)
    if not v.ok:
        record.error = "validation_failed"
        record.total_duration_s = time.time() - t_main
        log_path = append_invocation(record, log_dir=args.log_dir)
        print(f"\n[log] {log_path}  ({record.total_duration_s:.2f}s)", file=sys.stderr)
        return 2

    # 2. Parse expr (and optional target) into sympy
    ab_cls, sb_cls = _get_bracket_classes()
    # opaque_den stays an *undefined* Function (no auto-eval to 1/S) so str(expr) keeps the
    # marker for the engine; the verifier/count layers interpret opaque_den(S) as 1/S.
    local_dict = {"ab": ab_cls, "sb": sb_cls, "I": sp.I, "opaque_den": sp.Function("opaque_den")}
    expr = parse_expr(raw_input, local_dict, lenient=True)
    target = (parse_expr(atomized_target, local_dict, lenient=True)
              if atomized_target else None)

    # 3. Run algorithms
    algo_names = [s.strip() for s in args.algorithms.split(",") if s.strip()]
    # Opaque polynomial denominators are a gbfs-FAMILY capability — SFT/CDS would mangle the
    # opaque_den(S) marker, but every GBFS variant supports polynomial denominators.
    # Auto-route: if the (atomized) input carries an opaque pole, keep all selected gbfs-family
    # backends and drop only the non-gbfs ones with a note (matches webservice/app.py's router;
    # the old code collapsed to model_free_gbfs alone, silently dropping gbfs_c/d/e/f).
    if has_opaque_den(raw_input):
        kept = [a for a in algo_names if is_gbfs(a)]
        dropped = [a for a in algo_names if not is_gbfs(a)]
        if not kept:
            kept = ["model_free_gbfs"]
            print("[route] Polynomial denominators require a GBFS backend. Using model_free_gbfs.")
        if dropped:
            print(f"[route] Polynomial denominators require GBFS. Running {kept}. Skipping {dropped}.")
        algo_names = kept

    # Size route: if the expression exceeds the SFT/CDS size budget (checkpoint tensor
    # widths), those backends can't run, but the model-free gbfs family has no such limit.
    # Drop SFT/CDS and keep gbfs; if nothing gbfs remains, there is nothing to run.
    size_issues = [i for i in v.issues if i.tag in SIZE_LIMIT_TAGS]
    if size_issues:
        gbfs_kept = [a for a in algo_names if is_gbfs(a)]
        dropped = [a for a in algo_names if not is_gbfs(a)]
        for iss in size_issues:
            print(f"[route] {iss.tag}: {iss.message}")
        if not gbfs_kept:
            msg = ("The expression exceeds the SFT/CDS checkpoint input limits. "
                   "Select a GBFS backend, such as model_free_gbfs.")
            print(f"[error] {msg}")
            record.error = f"size_limit: {msg}"
            record.total_duration_s = time.time() - t_main
            log_path = append_invocation(record, log_dir=args.log_dir)
            print(f"[log] {log_path}", file=sys.stderr)
            return 2
        if dropped:
            print(f"[route] The expression exceeds the SFT/CDS checkpoint input limits. "
                  f"Running {gbfs_kept}. Skipping {dropped}.")
        algo_names = gbfs_kept

    # Engine route: a rational coefficient or the imaginary unit is accepted by the learned
    # backends but not representable by the search engine. Drop the search family with a
    # note; fail only when nothing is left to run.
    engine_issues = [i for i in v.issues if i.tag in ENGINE_LIMIT_TAGS]
    if engine_issues:
        kept = [a for a in algo_names if not is_gbfs(a)]
        dropped = [a for a in algo_names if is_gbfs(a)]
        for iss in engine_issues:
            print(f"[route] {iss.tag}: {iss.message}")
        if not kept:
            msg = ("The search engine cannot represent this input. See LIMIT_ENGINE_COEFF above. "
                   "Select an SFT or CDS backend.")
            print(f"[error] {msg}")
            record.error = f"engine_limit: {msg}"
            record.total_duration_s = time.time() - t_main
            log_path = append_invocation(record, log_dir=args.log_dir)
            print(f"[log] {log_path}", file=sys.stderr)
            return 2
        if dropped:
            print(f"[route] The search engine cannot represent this input. "
                  f"Running {kept}. Skipping {dropped}.")
        algo_names = kept

    unknown = [a for a in algo_names if not is_known(a)]
    if unknown:
        msg = f"Unknown backends: {unknown}. Available: {', '.join(list_algorithms())}."
        print(f"[error] {msg}")
        record.error = msg
        log_path = append_invocation(record, log_dir=args.log_dir)
        print(f"[log] {log_path}", file=sys.stderr)
        return 3

    results = run_all(
        algorithm_names=algo_names,
        expr=expr,
        n_point=v.inferred_n_point,
        target=target,
        max_steps=args.max_steps,
        per_step_timeout=args.per_step_timeout,
        sample_timeout=args.sample_timeout,
        capture_derivation=args.capture_derivation,
        simplify_mode=args.simplify_mode,
        sft_patience=args.sft_patience,
        sft_beam_size=args.sft_beam_size,
        sft_beam_temperature=args.sft_beam_temperature,
        cds_beam_size=args.cds_beam_size,
        cds_temperature=args.cds_temperature,
        cds_nucleus_p=args.cds_nucleus_p,
        cds_length_penalty=args.cds_length_penalty,
        gbfs_max_apps=args.gbfs_max_apps,
        gbfs_max_seconds=args.gbfs_max_seconds,
        gbfs_max_terms=args.gbfs_max_terms,
        gbfs_den_aware=args.gbfs_den_aware,
        gbfs_ordering=args.gbfs_ordering,
        gbfs_threads=args.gbfs_threads,
        gbfs_target_criterion=args.gbfs_target_criterion,
        gbfs_audit_steps=args.gbfs_audit_steps,
        gbfs_split_max_terms=args.gbfs_split_max_terms,
        gbfs_c_cancel_max_terms=args.gbfs_c_cancel_max_terms,
        post_timeout_s=args.post_process_timeout,
        algo_grace_s=args.algo_grace_seconds,
        max_parallel=args.max_parallel_algos,
        # ORIGINAL pre-atomize input (record.raw_input; the local `raw_input` was reassigned
        # to the atomized form above) — gbfs_e splits it by the original additive terms.
        raw_input=record.raw_input,
    )
    record.results = results

    # 4. Output
    _print_results(results)
    record.total_duration_s = time.time() - t_main
    log_path = append_invocation(record, log_dir=args.log_dir)
    print(f"\n[log] {log_path}  "
          f"(total={record.total_duration_s:.2f}s, "
          f"validate={record.validation_duration_s:.3f}s)")

    if args.json:
        print()
        print(json.dumps(record.to_dict(), default=str, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
