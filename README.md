# AmpLens

Tools for revealing physical structure in scattering amplitudes.

AmpLens is a collection of tools for scattering amplitudes in spinor-helicity notation. It currently includes AmpZip, a simplifier for massless amplitudes. The `amplens` package installs the available components together.

| Component | Directory | PyPI | What it does |
|---|---|---|---|
| AmpZip | [`AmpZip/`](AmpZip) | `ampzip` (engine: `amplens-engine`) | Simplifies four-, five- and six-point spinor-helicity expressions by greedy best-first search. |
| AmpLens | [`amplens/`](amplens) | `amplens` | Installs the components, lists their versions, and provides access to the AmpZip command-line interface. |

## Install

```bash
pip install amplens             # AmpLens components
pip install 'amplens[web]'       # plus the AmpZip web service
pip install ampzip              # AmpZip alone
```

<!-- Confirm the platform list against the artifacts of the first release before publishing. -->
Binary wheels are provided for CPython 3.12 and later on Linux (x86_64, aarch64), macOS (Apple silicon, Intel) and Windows (x86_64). Building from source requires a Rust toolchain. See the [AmpZip README](AmpZip/README.md) for instructions.

## Layout

```
AmpLens/
├── AmpZip/        the AmpZip package, its engine, tests and documentation
├── amplens/       the meta package
├── paper/         input datasets and evaluation commands for the paper
└── .github/       release workflow (tags ampzip-v* and amplens-v*)
```

## Citation

If you use AmpZip, please cite its [paper and software](AmpZip/CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
