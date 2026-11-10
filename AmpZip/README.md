# AmpZip

AmpZip simplifies massless four-, five- and six-point spinor-helicity amplitudes using greedy best-first search over Schouten, momentum-conservation and momentum-squared identities. It provides command-line and web interfaces, with optional SFT and CDS backends.

Input notation:

- `ab(i,j)` denotes an angle bracket ⟨i j⟩.
- `sb(i,j)` denotes a square bracket [i j].
- `s(i,j,...)` denotes the Mandelstam invariant s<sub>ij…</sub>.
- `chain('as', a, [m1, m2, ...], b)` denotes a Dirac chain ⟨a|(pₘ₁ + pₘ₂ + ⋯)|b].

Combine these using `+`, `-`, `*`, `/`, `**` and parentheses. For example:

```text
ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))
```

See [docs/LIMITS.md](docs/LIMITS.md) for details.

## Install

**From PyPI**, as binary wheels for Linux, macOS and Windows on Python 3.12 and later:

```bash
pip install ampzip            # command line
pip install 'ampzip[web]'     # plus the web service
```

**From source**, which needs a Rust toolchain from <https://rustup.rs>:

```bash
git clone https://github.com/AmpLens/AmpLens
cd AmpLens/AmpZip
pip install ./ampzip/engine          # the Rust engine, distributed as `ampzip-engine`
pip install -e '.[web,dev]'
python tests/run_tests.py
```

**Web service with Docker**, run from the `AmpZip` directory:

```bash
docker compose up             # then open http://127.0.0.1:8000
```

## Command line

```bash
ampzip --expr "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))"
ampzip --expr "ab(1,3)*sb(2,4)*(s(1,2)+s(1,4))/s(1,2)**2"
ampzip --expr "(chain('as',6,[1,2],3)**3)/(ab(6,1)*ab(1,2)*sb(3,4)*sb(4,5)*s(6,1,2))" \
       --algorithms model_free_gbfs,gbfs_f --gbfs-max-apps 60000 --gbfs-max-seconds 120
ampzip --expr-file amplitude.txt --json
```

The number of legs is inferred from the labels or can be specified by `--n-point`. For each backend, the output shows the bracket counts before and after simplification and the simplest expression found. AmpZip logs each simplification to `cli.jsonl` in the user log directory. Use `--log-dir` or `AMPZIP_LOG_DIR` to change the location.

Run `ampzip --help` to see all options.

Use `--algorithms` to select one or more backends, separated by commas.

| Backend | Description |
|---|---|
| `model_free_gbfs` | The default. Greedy best-first search with the factored bracket count as its objective. |
| `gbfs_c`, `gbfs_d`, `gbfs_e`, `gbfs_f` | Search variants with alternative objectives or per-term simplification. |
| `sft_greedy` | SFT with greedy action selection. |
| `sft_best_of_n` | SFT with best-of-N selection. |
| `cds_oneshot` | CDS sequence-to-sequence simplification. |

SFT uses the model checkpoints published with arXiv:2603.11164. CDS uses the model checkpoints published with arXiv:2408.04720.

SFT and CDS require separate installation. See [Optional learned backends](#optional-learned-backends) below.

## Web service

```bash
WEBSVC_AUTH_MODE=dev \
WEBSVC_DEV_ALLOW_NO_AUTH=1 \
ampzip-web  # http://127.0.0.1:8000
```

Enter an expression and select the backends to run. Results appear as each backend finishes. Available derivations are shown in the derivation panel.

The command above starts the service locally without authentication. For remote access, configure HTTP Basic or Cloudflare Access as described in the [deployment](docs/DEPLOY.md) and [security](docs/SECURITY.md) guides.

## Optional learned backends

The SFT and CDS backends require code and model checkpoints from their authors. Use the commands below to install the dependencies and download the required files.

```bash
pip install 'ampzip[sft]'
ampzip fetch-backends

pip install 'ampzip[cds]'
ampzip fetch-backends --cds  # CDS checkpoints, approximately 890 MB
```

If you downloaded the backends to a different directory, set `AMPZIP_BACKENDS` to that path.

## Documentation

- [Input notation and limits](docs/LIMITS.md)
- [GBFS variants](docs/GBFS_VARIANTS.md)
- [Rust engine and Python API](docs/ENGINE.md)
- [Web service deployment](docs/DEPLOY.md)
- [Web service security](docs/SECURITY.md)
- [Known issues](docs/KNOWN_ISSUES.md)
- [Changelog](CHANGELOG.md)
- [Citation information](CITATION.cff)

Inputs and commands for the paper's experiments are available in [Paper resources](https://github.com/AmpLens/AmpLens/tree/main/paper).

## Paper and citation

If you use AmpZip or any scientific part of its code in your research, please cite:

Yifan Fei and Zhen Liu, **Fast, training-free simplification of spinor-helicity amplitudes**, [arXiv:2611.10115](https://arxiv.org/abs/2611.10115) [hep-th] (2026).

```bibtex
@article{Fei:2026gwf,
    author = "Fei, Yifan and Liu, Zhen",
    title = "{Fast, training-free simplification of spinor-helicity amplitudes}",
    eprint = "2611.10115",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "UMN-TH-4603/26",
    month = "11",
    year = "2026"
}
```

## License

MIT, see [LICENSE](LICENSE).
