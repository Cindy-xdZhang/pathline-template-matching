"""Run rank-likelihood tests with only the Python standard library.

Ibex's frozen ``deepvortex`` environment intentionally has no pytest package.
This runner supports the sole fixture used by this bounded test set,
``tmp_path``, and fails closed if any new fixture is introduced.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

MODULE_NAMES = (
    "test_source_centered_rank_likelihood",
    "test_source_centered_rank_likelihood_runner",
    "test_source_centered_rank_likelihood_aggregate",
    "test_source_centered_rank_likelihood_ibex",
)


def _case(function):
    parameters = tuple(inspect.signature(function).parameters)
    if not parameters:
        return unittest.FunctionTestCase(function)
    if parameters == ("tmp_path",):
        def invoke_with_tmp_path() -> None:
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))

        invoke_with_tmp_path.__name__ = function.__name__
        return unittest.FunctionTestCase(invoke_with_tmp_path)
    raise AssertionError(
        f"unsupported test fixture contract for {function.__module__}."
        f"{function.__name__}: {parameters}"
    )


def load_tests(loader, standard_tests, pattern):
    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    count = 0
    for module_name in MODULE_NAMES:
        module = importlib.import_module(module_name)
        test_case_classes = sorted(
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and issubclass(value, unittest.TestCase)
            and value is not unittest.TestCase
        )
        if test_case_classes:
            raise AssertionError(
                f"unsupported TestCase classes in {module_name}: {test_case_classes}"
            )
        functions = sorted(
            (
                value
                for name, value in vars(module).items()
                if name.startswith("test_") and inspect.isfunction(value)
            ),
            key=lambda value: value.__name__,
        )
        if not functions:
            raise AssertionError(f"no tests discovered in {module_name}")
        for function in functions:
            suite.addTest(_case(function))
            count += 1
    if count != 36:
        raise AssertionError(f"expected exactly 36 bounded tests, found {count}")
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
