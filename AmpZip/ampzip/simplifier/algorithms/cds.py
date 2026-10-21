"""CDS backend — Dersy et al. seq2seq simplifier (arXiv 2408.04720).

Wraps spinorhelicity's `model.simplifier_methods.one_shot_simplify` — a single
encoder-decoder pass with token-level beam search (or nucleus sampling). This is
the same model + decoding mode that `src/eval_paper_model_{n}pt.py` runs over the
test set (98.01% on 4pt). The iterative term-grouping mode
(`contrastive_simplifier.total_simplification`) is intentionally not used here —
it requires a separately-trained contrastive encoder and is only worthwhile for
expressions too long to fit the seq2seq context window (2560 tokens).

State: env + (encoder, decoder) are lazily constructed per n_point and cached.
Each n_point load is ~300 MB of model weights.
"""

from __future__ import annotations
import os
import sys
import threading
import time
import types
from typing import Dict, Optional

import sympy as sp
import torch

from .. import _paths
from ..result import AlgorithmResult
from ._post import count_brackets as _count_brackets, maybe_simplify

_paths.assert_rl_repo_present()  # the upstream packages are fetched, not shipped


DEFAULT_BEAM_SIZE = 5
DEFAULT_TEMPERATURE = 1.5
DEFAULT_NUCLEUS_P = 0.0      # 0 = pure beam; >0 = nucleus sampling at this p
DEFAULT_LENGTH_PENALTY = 1.0

# Shipped CDS checkpoints (under spinorhelicity/)
_MODEL_FILE_PER_NPT = {
    4: "n4_i1_3_l1000_10Mex.pth",
    5: "n5_simplifier.pth",
    6: "n6_simplifier.pth",
}

# Max input sequence length the encoder can handle (n_max_positions).
CDS_MAX_INPUT_TOKENS = 2560


def _ensure_apex_mock():
    """spinorhelicity imports `apex` even when not training — mock it."""
    if "apex" in sys.modules:
        return
    apex = types.ModuleType("apex")
    apex.parallel = types.ModuleType("apex.parallel")
    apex.amp = types.ModuleType("apex.amp")
    sys.modules["apex"] = apex
    sys.modules["apex.parallel"] = apex.parallel
    sys.modules["apex.amp"] = apex.amp


def _build_params(n_point: int, use_cuda: bool):
    """Mirror `src/eval_paper_model_{n}pt.py` parameters dict.

    Most fields are required by `build_env` / `build_modules` even though we
    bypass training. The model-architecture fields (emb_dim, n_*_layers, n_heads,
    n_max_positions) MUST match what the shipped checkpoint expects.
    """
    _ensure_apex_mock()
    from environment.utils import AttrDict

    REPO_ROOT = str(_paths.RL_REPO)
    model_file = _MODEL_FILE_PER_NPT[n_point]
    data_dir = f"n{n_point}_i1_3"

    return AttrDict({
        "exp_name": f"cds_cli_n{n_point}",
        "dump_path": "/tmp",
        "exp_id": "inline",
        "save_periodic": 0,
        "tasks": "spin_hel",

        # environment
        "env_name": "char_env",
        "npt_list": [n_point],
        "max_scale": 2,
        "max_terms": 3,
        "max_scrambles": 3,
        "min_scrambles": 1,
        "save_info_scr": False,
        "save_info_scaling": False,
        "int_base": 10,
        "numeral_decomp": True,
        "max_len": CDS_MAX_INPUT_TOKENS,
        "l_scale": 0.75,
        "numerator_only": True,
        "reduced_voc": True,
        "all_momenta": False,

        # model (must match checkpoint exactly)
        "emb_dim": 512,
        "n_enc_layers": 3,
        "n_dec_layers": 3,
        "n_heads": 8,
        "dropout": 0,
        "n_max_positions": CDS_MAX_INPUT_TOKENS,
        "attention_dropout": 0,
        "sinusoidal_embeddings": False,
        "share_inout_emb": True,
        "positional_encoding": True,
        "reload_model": os.path.join(REPO_ROOT, "spinorhelicity", model_file),

        # trainer (build_env / build_modules need these even though we don't train)
        "export_data": False,
        "reload_data": "spin_hel,{t},{v},{e}".format(
            t=os.path.join(REPO_ROOT, "spinorhelicity", "data", data_dir, "data.prefix.counts.train_1000"),
            v=os.path.join(REPO_ROOT, "spinorhelicity", "data", data_dir, "data.prefix.counts.valid_1000"),
            e=os.path.join(REPO_ROOT, "spinorhelicity", "data", data_dir, "data.prefix.counts.test_1000"),
        ),
        "reload_size": "",
        "epoch_size": 1000,
        "max_epoch": 1,
        "amp": -1,
        "fp16": False,
        "accumulate_gradients": 1,
        "optimizer": "adam,lr=0.0001",
        "clip_grad_norm": 5,
        "stopping_criterion": "",
        "validation_metrics": "",
        "reload_checkpoint": "",
        "env_base_seed": -1,
        "batch_size": 32,

        # evaluation defaults (per-call overrides handled below)
        "eval_only": True,
        "test_file": True,
        "valid_file": False,
        "numerical_check": 2,
        "eval_verbose": 0,
        "eval_verbose_print": False,
        "beam_eval": True,
        "beam_size": 5,
        "beam_length_penalty": 1,
        "beam_early_stopping": True,
        "nucleus_sampling": False,
        "nucleus_p": 0.95,
        "temperature": 1.5,
        "scaling_eval": False,

        # SLURM/GPU
        "cpu": not use_cuda,
        "local_rank": -1,
        "master_port": -1,
        "num_workers": 1,
        "debug_slurm": False,
        "lib_path": "",
        "mma_path": None,
    })


class _LoadedCDSBackend:
    """One per n_point — env + (encoder, decoder) + ab/sb classes for re-parsing."""

    def __init__(self, n_point: int, use_cuda: bool):
        _ensure_apex_mock()
        from environment import build_env
        from model import build_modules
        import environment as env_mod

        params = _build_params(n_point, use_cuda)
        env_mod.utils.CUDA = not params.cpu

        self.params = params
        self.env = build_env(params)
        modules = build_modules(self.env, params)
        self.encoder = modules["encoder"]
        self.decoder = modules["decoder"]
        self.encoder.eval()
        self.decoder.eval()
        self.n_point = n_point


_CDS_BACKENDS: Dict[int, _LoadedCDSBackend] = {}


def _get_cds_backend(n_point: int, use_cuda: bool) -> _LoadedCDSBackend:
    if n_point not in _CDS_BACKENDS:
        _CDS_BACKENDS[n_point] = _LoadedCDSBackend(n_point, use_cuda)
    return _CDS_BACKENDS[n_point]


def _reparse_in_env(expr, env):
    """Re-parse a sympy expression so its ab/sb atoms are the env's classes.

    SFT-side ab/sb (from `environment.spin_helicity_env`) and CDS-side ab/sb
    (from `environment.bracket_env`, registered in env.func_dict) are different
    sympy Function classes. one_shot_simplify needs the env's flavor.

    Also calls `reorder_expr` to match the canonical ordering streamlit uses
    via `simplifier_methods.load_equation` — the tokenizer is sensitive to
    operand order on `Add` / `Mul`.
    """
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication,
    )
    from environment.utils import reorder_expr

    s = str(expr)
    try:
        out = parse_expr(s, local_dict=env.func_dict)
    except SyntaxError:
        out = parse_expr(s, local_dict=env.func_dict,
                         transformations=standard_transformations + (implicit_multiplication,))
    return reorder_expr(out)


def _one_shot_simplify(envir, module_transfo, f_eq, params_in, length_penalty=1.0):
    """One encoder pass and one decode of `f_eq`, returning `[(matches, hyp, rel_diff)]`.

    This is the upstream `model.simplifier_methods.one_shot_simplify` with the beam length
    penalty exposed: upstream hard-codes `length_penalty=1`, which made `--cds-length-penalty`
    an inert parameter. Everything else (encoding, greedy / beam / nucleus decoding, parsing
    and the numerical check of each hypothesis) is the same code path, imported from upstream.
    """
    import torch
    from model import simplifier_methods as _sm
    encoder, decoder = module_transfo
    x1, len1 = _sm.one_hot_encode_sp(f_eq, envir)
    if len1.item() > encoder.n_max_positions:
        raise IndexError("expression exceeds the CDS checkpoint input length limit")
    x1, len1 = _sm.to_cuda(x1, len1)
    f_sp = _sm.revert_sp_form(f_eq)
    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
    beam_sz, sample_method, nucleus_prob, temp = params_in
    with torch.no_grad():
        if sample_method == "Greedy Decoding":
            greedy_sol, _ = decoder.generate(encoded.transpose(0, 1), src_len=len1, max_len=2048,
                                             sample_temperature=None)
            hypotheses = [(0.0, greedy_sol[:-1].squeeze())]
        else:
            nucleus_sample = sample_method == "Nucleus Sampling"
            _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_sz,
                                               length_penalty=float(length_penalty),
                                               early_stopping=True, max_len=2048,
                                               stochastic=nucleus_sample, nucl_p=nucleus_prob,
                                               temperature=temp, rng_gen=None)
            assert len(beam) == 1
            hypotheses = beam[0].hyp
    out_hyp = []
    for score, sent in sorted(hypotheses, key=lambda y: y[0], reverse=True):
        ids = sent[1:].tolist()
        tok = [envir.id2word[wid] for wid in ids]
        try:
            hyp = envir.prefix_to_infix(tok)
            hyp = envir.infix_to_sympy(hyp)
            hyp_disp = _sm.convert_sp_forms(hyp, envir.func_dict)
            npt = envir.npt_list[0] if len(envir.npt_list) == 1 else None
            matches, rel_diff = _sm.check_numerical_equiv_local(envir.special_tokens, hyp, f_sp, npt=npt)
            out_hyp.append((matches, hyp_disp, rel_diff))
        except Exception:  # noqa: BLE001 - an unparsable hypothesis is skipped, as upstream does
            pass
    return list(dict.fromkeys(out_hyp))


class CDS:
    name = "cds_oneshot"

    def __init__(
        self,
        beam_size: int = DEFAULT_BEAM_SIZE,
        temperature: float = DEFAULT_TEMPERATURE,
        nucleus_p: float = DEFAULT_NUCLEUS_P,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        device: Optional[str] = None,
    ):
        self.beam_size = beam_size
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self.length_penalty = length_penalty
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _sample_method(self) -> str:
        if self.nucleus_p > 0.0:
            return "Nucleus Sampling"
        if self.beam_size == 1:
            return "Greedy Decoding"
        return "Beam Search"

    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,
        per_step_timeout: float = 30.0,
        sample_timeout: float = 120.0,
        capture_derivation: bool = False,
        simplify_mode: str = "none",
        **kwargs,   # absorb orchestrator kwargs CDS ignores (sft_patience, cancel_event, …)
    ) -> AlgorithmResult:
        start_brackets = _count_brackets(expr)
        t0 = time.time()

        try:
            backend = _get_cds_backend(n_point, use_cuda=(self.device == "cuda"))
            env_expr = _reparse_in_env(expr, backend.env)

            params_in = (
                self.beam_size,
                self._sample_method(),
                self.nucleus_p if self.nucleus_p > 0 else 0.95,
                self.temperature,
            )
            # The decode is one forward pass plus beam search and cannot be interrupted. It is
            # waited for at most `sample_timeout` seconds: on expiry the result reports a
            # timeout and the decode runs on in its thread until it finishes (output discarded).
            # So `sample_timeout` bounds the wait, not the computation; see docs/LIMITS.md.
            box: dict = {}

            def _decode():
                try:
                    with torch.no_grad():
                        box["hyp"] = _one_shot_simplify(
                            backend.env,
                            (backend.encoder, backend.decoder),
                            env_expr,
                            params_in,
                            length_penalty=self.length_penalty,
                        )
                except BaseException as exc:  # noqa: BLE001  (re-raised in the caller)
                    box["exc"] = exc

            # Not a daemon thread: a torch decode still running when the interpreter exits
            # aborts the process ("terminate called without an active exception"), so a CLI
            # call that timed out prints its result and then exits once the decode is done.
            th = threading.Thread(target=_decode, name="cds-decode", daemon=False)
            th.start()
            th.join(sample_timeout if sample_timeout and sample_timeout > 0 else None)
            if th.is_alive():
                return AlgorithmResult(
                    algorithm=self.name,
                    final_expr_str="",            # no form reported: nothing to count or judge
                    bracket_count_start=start_brackets,
                    bracket_count_final=-1,
                    bracket_count_min=-1,
                    n_steps=0,
                    solved=False,
                    timed_out=True,
                    duration_s=time.time() - t0,
                    extra={
                        "stop_reason": "time",
                        "sample_timeout_s": sample_timeout,
                        "decode_still_running": True,
                        "note": "the decode exceeded sample_timeout; it cannot be interrupted and "
                                "runs on in the background until it finishes (its output is discarded)",
                        "beam_size": self.beam_size,
                        "sample_method": self._sample_method(),
                    },
                )
            if "exc" in box:
                raise box["exc"]
            hypotheses = box["hyp"]
        except (AssertionError, IndexError) as e:
            return AlgorithmResult(
                algorithm=self.name,
                final_expr_str=str(expr),
                bracket_count_start=start_brackets,
                bracket_count_final=start_brackets,
                bracket_count_min=start_brackets,
                n_steps=0,
                solved=False,
                timed_out=False,
                duration_s=time.time() - t0,
                extra={
                    "error": f"{type(e).__name__}: {e}",
                    "beam_size": self.beam_size,
                    "sample_method": self._sample_method(),
                },
            )

        duration = time.time() - t0

        # Rank candidates: numerical match first, then lowest bracket count.
        scored = []
        for (matches, hyp_sp, rel_diff) in hypotheses:
            scored.append({
                "matches": bool(matches),
                "expr": hyp_sp,
                "bracket_count": _count_brackets(hyp_sp),
                "rel_diff": rel_diff,
            })

        matching = [c for c in scored if c["matches"]]
        if matching:
            best = min(matching, key=lambda c: c["bracket_count"])
            solved = best["bracket_count"] < start_brackets
        elif scored:
            best = min(scored, key=lambda c: c["bracket_count"])
            solved = False
        else:
            return AlgorithmResult(
                algorithm=self.name,
                final_expr_str=str(expr),
                bracket_count_start=start_brackets,
                bracket_count_final=start_brackets,
                bracket_count_min=start_brackets,
                n_steps=0,
                solved=False,
                timed_out=False,
                duration_s=duration,
                extra={
                    "error": "no parseable candidate from decoding",
                    "beam_size": self.beam_size,
                    "sample_method": self._sample_method(),
                },
            )

        raw_final_str = str(best["expr"])
        final_str, bk_final, _ = maybe_simplify(best["expr"], simplify_mode)
        if bk_final < 0:
            bk_final = best["bracket_count"]
            final_str = raw_final_str

        # Re-evaluate solved against post-simplify bracket count.
        solved = best["matches"] and bk_final < start_brackets

        candidate_summaries = [
            {
                "matches": c["matches"],
                "bracket_count": c["bracket_count"],
                "rel_diff": (str(c["rel_diff"]) if c["rel_diff"] is not None else None),
                "expr": str(c["expr"]),
            }
            for c in scored
        ]

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=final_str,         # CDS is single-step: min == final
            bracket_count_start=start_brackets,
            bracket_count_final=bk_final,
            bracket_count_min=bk_final,
            n_steps=1,
            solved=solved,
            timed_out=False,
            duration_s=duration,
            extra={
                "beam_size": self.beam_size,
                "sample_method": self._sample_method(),
                "temperature": self.temperature,
                "nucleus_p": self.nucleus_p,
                "length_penalty": self.length_penalty,
                "all_candidates": candidate_summaries,
                "numerical_match": best["matches"],
                "rel_diff": (str(best["rel_diff"]) if best["rel_diff"] is not None else None),
                "raw_final_expr": raw_final_str,
                "raw_bracket_count_final": best["bracket_count"],
                "raw_bracket_count_min": best["bracket_count"],
                "simplify_mode": simplify_mode,
            },
        )
