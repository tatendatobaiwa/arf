"""The defaults release 2 assumes for a setting a manifest does not state are those of the registered Cell 0."""

import ast

import pytest
from registered import RELEASES

from arf.runs import RELEASE_DEFAULTS


@pytest.mark.parametrize("release", list(RELEASES))
def test_every_default_is_the_registered_configuration_default(release):
    tree = ast.parse(RELEASES[release].read_text())
    (config,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ARFConfig"]
    defaults = {
        field.target.id: field.value
        for field in config.body
        if isinstance(field, ast.AnnAssign) and field.value is not None
    }
    for name, ours in RELEASE_DEFAULTS.items():
        theirs = ast.literal_eval(defaults[name])  # a KeyError means the setting is not a field of Cell 0
        assert (list(theirs) if isinstance(theirs, tuple) else theirs) == ours, name
