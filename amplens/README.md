# amplens

This package installs the components of [AmpLens](https://github.com/AmpLens/AmpLens). It currently includes AmpZip (`ampzip`) and its Rust engine (`ampzip-engine`).

```bash
pip install amplens
pip install 'amplens[web]'     # plus the AmpZip web service
amplens components           # list installed components and versions
```

Pass AmpZip options directly to `amplens`:

```bash
amplens --expr "(ab(1,2)*ab(3,4)-ab(1,3)*ab(2,4))/ab(1,4)"
```

This expression simplifies to `-ab(2,3)` by the Schouten identity.

You can also run the same commands with `python -m amplens`.

See the [AmpZip documentation](https://github.com/AmpLens/AmpLens/tree/main/AmpZip) for input notation, command-line options and web service setup.
