"""Placeholder test confirming the pytest harness runs and the package imports."""


def test_harness_runs():
    assert True


def test_src_package_importable():
    import src  # noqa: F401
    import src.agents  # noqa: F401
    import src.tools  # noqa: F401
    import src.models  # noqa: F401
    import src.state  # noqa: F401
    import src.prompts  # noqa: F401
