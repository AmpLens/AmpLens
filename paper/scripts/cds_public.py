"""Run CDS on the public test sets with the inference settings used in the paper.

    python paper/scripts/cds_public.py --n 4 --out cds_4.jsonl
    python paper/scripts/cds_public.py --n 6 --start 0 --end 500 --gpu 1 --out part0.jsonl
    python paper/scripts/cds_public.py --suite-case 6 30 120 144 --out case144.json

Requires `pip install 'ampzip[cds]'` and `ampzip fetch-backends --cds --cds-data`. Nothing from the upstream repositories is copied here: the script loads the authors' evaluation module `src/eval_paper_model_{n}pt.py`, reads its parameter dictionary, sets the beam size to 20 and the batch size to 1, and replaces only the evaluation loop.

Each case is one call from the input expression string to the returned expression string:

1. Decode greedily. If the output is token for token equal to the target, the case is solved.
2. Otherwise generate 20 beam candidates and check them in score order with the authors' equivalence check. The case is solved by the first candidate that is equivalent to the input and meets the success criterion of the paper.

The criterion is the one applied to GBFS: in the normal form of the AmpZip engine, the raw numerator bracket count and the number of numerator terms are at most those of the target. The authors' own size test also counts denominator brackets. The paper's timings use their stopping rule, and its solve decisions apply the paper's criterion to all 20 candidates; stopping at the first candidate that meets the criterion gives the same decisions. One JSON line is written per case and the solve rate is printed at the end.

`--suite-case N K L IDX` runs one case of the size-by-depth suite instead, with the suite criterion (raw numerator bracket count only). It writes one JSON object that also lists every beam candidate with its raw numerator bracket count, its number of terms and the outcome of the finite-field check against the input, as in the comparison table for case 144 in the appendix.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import importlib.util
import inspect
import json
import math
import os
import random
import re
import textwrap
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "inputs", "size_depth_suite.jsonl.gz")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--n", type=int, choices=[4, 5, 6], help="public test set")
    which.add_argument("--suite-case", type=int, nargs=4, metavar=("N", "K", "L", "IDX"),
                       help="one case of the size-by-depth suite")
    ap.add_argument("--suite", default=SUITE, help="suite file for --suite-case")
    ap.add_argument("--start", type=int, default=0, help="first case index")
    ap.add_argument("--end", type=int, default=0, help="end of the case range (0 = all)")
    ap.add_argument("--gpu", default="0", help="CUDA device index; use --cpu to run without a GPU")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True, help="JSON lines, one per case")
    return ap.parse_args()


def load_parameters(path, repo):
    """The parameter dictionary of an upstream evaluation script, evaluated with its REPO_ROOT."""
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and getattr(n.targets[0], "id", None) == "parameters")
    return eval(compile(ast.Expression(node.value.args[0]), str(path), "eval"),
                {"os": os, "REPO_ROOT": str(repo)})


def install_generate_compatibility(model_class):
    """Current PyTorch requires a boolean mask where the upstream greedy decoder builds a byte mask at the length cap. Swap that one call at run time; decoding is otherwise unchanged."""
    source = textwrap.dedent(inspect.getsource(model_class.generate))
    if source.count("unfinished_sents.byte()") != 1:
        raise SystemExit(
            "the upstream greedy decoder changed; the compatibility patch does not apply")
    namespace = dict(model_class.generate.__globals__)
    exec(compile(source.replace("unfinished_sents.byte()", "unfinished_sents.bool()"),
                 "<cds-greedy-bool-mask>", "exec"), namespace)
    model_class.generate = namespace["generate"]


def load_suite_case(path, n, K, L, idx):
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if (r["n"], r["K"], r["L"], r["idx"]) == (n, K, L, idx):
                return r
    raise SystemExit(f"no suite case n={n}, K={K}, L={L}, idx={idx} in {path}")


def main():
    a = parse_args()
    a.out = os.path.abspath(a.out)           # the upstream module runs from its own directory
    n = a.n or a.suite_case[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.cpu else a.gpu
    from ampzip.simplifier import _paths
    repo = _paths.ensure_importable()
    # training/optim.py calls inspect.getargspec, removed in Python 3.11; same argument list.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    src = repo / "src" / f"eval_paper_model_{n}pt.py"
    test = repo / "spinorhelicity" / "data" / f"n{n}_i1_3" / "data.prefix.counts.test_1000"
    if not test.exists():
        raise SystemExit(f"{test} missing: run `ampzip fetch-backends --cds --cds-data`")
    lines = test.read_text().splitlines()
    indices = range(a.start, a.end or len(lines))
    case = load_suite_case(a.suite, *a.suite_case) if a.suite_case else None
    params = load_parameters(src, repo)
    workdir = a.out + ".upstream"
    params.update(cpu=a.cpu, batch_size=1, beam_size=20, eval_verbose=0,
                  dump_path=workdir, exp_name=f"cds_public_{n}pt",
                  exp_id=f"{a.start}_{a.end}", reload_data=f"spin_hel,{test},{test},{test}")

    spec = importlib.util.spec_from_file_location(f"cds_eval_{n}pt", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # module level only: its __main__ block is skipped
    import logging
    import numpy as np
    import torch
    import sympy as sp
    from model.transformer import TransformerModel
    from training.evaluator import Evaluator, check_hypothesis, idx_to_sp
    from environment.utils import TimeoutError as CheckTimeout, convert_sp_forms
    import ampzip_engine as E
    from ampzip.verify import equal
    torch.set_num_threads(1)
    install_generate_compatibility(TransformerModel)

    bracket = re.compile(r"\b(ab|sb)(\d)(\d)\b")
    ab, sb = sp.Function("ab"), sp.Function("sb")

    def to_expr(text):
        """Parse the evaluator's compact `ab12` notation."""
        return sp.sympify(bracket.sub(lambda m: f"{m[1]}({m[2]},{m[3]})", text),
                          locals={"ab": ab, "sb": sb})

    def size(expr):
        """Raw numerator bracket count and numerator term count in the engine's normal form."""
        e = E.Expr(str(expr))
        return e.bracket_count(), e.num_terms()

    @torch.no_grad()
    def evaluate(evaluator):
        env, p = evaluator.env, evaluator.params
        encoder, decoder = evaluator.modules["encoder"], evaluator.modules["decoder"]
        encoder.eval()
        decoder.eval()
        device = next(encoder.parameters()).device
        logging.getLogger().setLevel(logging.CRITICAL)
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

        def solve(text, reference_ids, limits):
            """One call from the input string to the returned expression. Also returns the beam candidates in score order (empty when the greedy output decided the case)."""
            sync()
            start = time.perf_counter()
            source = sp.sympify(text, locals=env.func_dict)
            source_ids = [env.word2id[w] for w in env.sympy_to_prefix(source)]
            x, lengths = env.batch_sequences([torch.LongTensor(source_ids)])
            x, lengths = x.to(device), lengths.to(device)
            encoded = encoder("fwd", x=x, lengths=lengths, causal=False)
            generated, generated_length = decoder.generate(
                encoded.transpose(0, 1), lengths, max_len=p.max_len)
            tokens = generated[:int(generated_length[0]), 0].tolist()
            row = dict(greedy_exact=tokens == [env.eos_index, *reference_ids, env.eos_index],
                       solved=False, beam_rank=None, output=None)
            hypotheses = []
            if row["greedy_exact"]:
                row.update(
                    solved=True,
                    output=str(convert_sp_forms(idx_to_sp(env, tokens[1:-1]), env.func_dict)))
            else:
                _, _, beams = decoder.generate_beam(
                    encoded.transpose(0, 1), lengths, beam_size=p.beam_size,
                    length_penalty=p.beam_length_penalty, early_stopping=p.beam_early_stopping,
                    max_len=p.max_len, stochastic=p.nucleus_sampling,
                    nucl_p=p.nucleus_p, temperature=p.temperature)
                hypotheses = sorted(beams[0].hyp, key=lambda h: h[0], reverse=True)
                best = None
                for rank, (score, hyp) in enumerate(hypotheses, 1):
                    eq = dict(i=0, j=0, score=float(score), src=source_ids,
                              tgt=reference_ids, hyp=hyp[1:].tolist())
                    try:
                        checked = check_hypothesis(eq, evaluator.session)
                    except CheckTimeout:
                        continue
                    if not checked["is_valid"]:
                        continue
                    expr = to_expr(checked["hyp"])
                    counts = size(expr)
                    if best is None or counts < best[0]:
                        best = (counts, str(expr))
                    if counts[0] <= limits[0] and counts[1] <= limits[1]:
                        row.update(solved=True, beam_rank=rank, output=str(expr))
                        break
                if not row["solved"]:
                    # The smallest equivalent candidate, or the input when there is none.
                    row["output"] = best[1] if best else str(source)
            sync()
            row["wall_seconds"] = round(time.perf_counter() - start, 4)
            return row, hypotheses

        if case is not None:
            target = sp.sympify(case["simple"], locals=env.func_dict)
            reference_ids = [env.word2id[w] for w in env.sympy_to_prefix(target)]
            # Suite criterion: raw numerator bracket count, no limit on the number of terms.
            limits = (E.Expr(case["simple"]).bracket_count(), math.inf)
            row, hypotheses = solve(case["scrambled"], reference_ids, limits)
            row.update(n=n, K=case["K"], L=case["L"], idx=case["idx"], target_brackets=limits[0])
            row["candidates"] = []
            for rank, (score, hyp) in enumerate(hypotheses, 1):
                item = dict(rank=rank, score=float(score))
                try:
                    cand = E.Expr(str(to_expr(str(convert_sp_forms(
                        idx_to_sp(env, hyp[1:].tolist()), env.func_dict)))))
                    item.update(expression=cand.to_str(), raw_brackets=cand.bracket_count(),
                                terms=cand.num_terms(),
                                equals_input=equal(cand.to_str(), case["scrambled"], n,
                                                   rng=random.Random(f"candidate:{rank}")))
                except Exception as e:  # noqa: BLE001 - an unparsable candidate
                    item.update(error=f"{type(e).__name__}: {e}")
                row["candidates"].append(item)
            with open(a.out, "w") as out:
                out.write(json.dumps(row, indent=2, default=str) + "\n")
            top = row["candidates"][0] if row["candidates"] else {}
            passing = sum(c.get("equals_input") is True for c in row["candidates"])
            print(f"CDS suite case n={n} K={case['K']} L={case['L']} idx={case['idx']}: "
                  f"solved={row['solved']}; highest-scoring candidate has "
                  f"{top.get('raw_brackets')} raw brackets; {passing} of "
                  f"{len(row['candidates'])} candidates pass the finite-field check; "
                  f"details in {a.out}")
            return {}

        solved = 0
        with open(a.out, "w") as out:
            for idx in indices:
                source_text, reference_text = lines[idx].split("|", 1)[1].split("\t")
                reference_ids = [env.word2id[t] for t in reference_text.split()]
                limits = size(to_expr(str(idx_to_sp(env, reference_ids))))
                # The call receives the input in ordinary ab(i,j)/sb(i,j) notation and tokenizes it.
                text = bracket.sub(lambda m: f"{m[1]}({m[2]},{m[3]})",
                                   env.prefix_to_infix(source_text.split()))
                row, _ = solve(text, reference_ids, limits)
                row = dict(idx=idx, **row)
                solved += row["solved"]
                out.write(json.dumps(row) + "\n")
                out.flush()
        print(f"CDS n={n}: solved {solved}/{len(indices)} = {100 * solved / len(indices):.2f}%; "
              f"per-case results in {a.out}")
        return {}

    Evaluator.run_all_evals = evaluate
    np.random.seed(42)
    torch.manual_seed(42)
    os.chdir(repo)
    p = mod.AttrDict(params)
    mod.check_model_params(p)
    mod.main(p)


if __name__ == "__main__":
    main()
