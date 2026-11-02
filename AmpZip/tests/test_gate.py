"""Real pytest gate over the existing check-suites.

The suites (test_atomize / test_reconstruct / test_gbfs_f) accumulate failures into a module
`_fails` list and return 1 from main() if non-empty, else 0. Historically pytest collected zero
items from them (assertions live in main()), so `pytest` was vacuously
green. These wrappers make the gate REAL: pytest goes RED if any suite reports a failure. This
file does not modify the suites' own logic.
"""
import test_atomize
import test_reconstruct
import test_gbfs_f


def test_atomize_suite():
    assert test_atomize.main() == 0, "atomize suite reported failures (see stdout)"


def test_reconstruct_suite():
    assert test_reconstruct.main() == 0, "reconstruct suite reported failures (see stdout)"


def test_gbfs_f_suite():
    assert test_gbfs_f.main() == 0, "gbfs_f suite reported failures (see stdout)"
