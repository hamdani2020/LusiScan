"""Unit tests for the scoped refactor engine (task 7.5).

Covers the ``libcst``-based transforms on fixture sources (the pydantic
``BaseSettings`` import move and the ``@validator`` / ``.dict()`` renames), the
manifest version bump, AST validation / invalid-AST rejection, and the executor
``apply_migration`` boundary including the ``guided_pr`` safe-fallback path.

Network is never touched here — the refactor engine is pure text/AST work.

_Requirements: 3.1 (AST-aware transform, not string replacement), 3.2 (manifest
version bump), 3.3 (leave non-auto-fixable changes flagged), 3.4 (transformed
source stays valid Python, verified by parsing), 3.5 (fall back to guided_pr
instead of committing broken code), 8.4 (validate model-generated code by AST
parsing before use)._
"""

from __future__ import annotations

import ast

import pytest

from src.tools import refactor_tools as rt


# --- validate_python_source (R3.4 / R8.4) ---------------------------------


class TestValidatePythonSource:
    def test_accepts_valid_source(self) -> None:
        assert rt.validate_python_source("x = 1\n") is True

    def test_accepts_empty_source(self) -> None:
        assert rt.validate_python_source("") is True

    def test_rejects_syntax_error(self) -> None:
        # A dangling ``def`` is not parseable.
        assert rt.validate_python_source("def broken(:\n") is False

    def test_rejects_unbalanced_parens(self) -> None:
        assert rt.validate_python_source("value = (1 + \n") is False

    def test_rejects_null_byte_source(self) -> None:
        # Embedded NUL raises ValueError from ast.parse, which we treat as invalid.
        assert rt.validate_python_source("x = 1\x00\n") is False

    def test_rejects_model_generated_garbage(self) -> None:
        # R8.4: model-generated code that doesn't parse must be rejected.
        model_output = "here is your code:\n    def f(: pass"
        assert rt.validate_python_source(model_output) is False


# --- normalize_name -------------------------------------------------------


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Pydantic", "pydantic"),
            ("Requests", "requests"),
            ("pydantic_settings", "pydantic-settings"),
            ("Flask-SQLAlchemy", "flask-sqlalchemy"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert rt.normalize_name(raw) == expected


# --- Pattern registry (task 7.1) ------------------------------------------


class TestPatternRegistry:
    def test_pydantic_is_registered(self) -> None:
        patterns = rt.get_patterns("pydantic")
        ids = {p.pattern_id for p in patterns}
        assert "pydantic.basesettings_import_move" in ids
        assert "pydantic.validator_rename" in ids
        assert "pydantic.dict_to_model_dump" in ids

    def test_requests_is_version_bump_only(self) -> None:
        # requests has no code transforms (version bump only).
        assert rt.get_patterns("requests") == []

    def test_pydantic_config_pattern_is_flagged_not_autofixable(self) -> None:
        patterns = {p.pattern_id: p for p in rt.get_patterns("pydantic")}
        config = patterns["pydantic.config_class_to_model_config"]
        assert config.auto_fixable is False
        assert config.build is None

    def test_unregistered_package_has_no_patterns(self) -> None:
        assert rt.get_patterns("flask") == []

    def test_lookup_is_name_normalized(self) -> None:
        assert rt.get_patterns("Pydantic") == rt.get_patterns("pydantic")


# --- pydantic transforms on fixtures (R3.1, R3.4) -------------------------


class TestPydanticImportMove:
    def test_single_basesettings_import_moves_module(self) -> None:
        result = rt.apply_source_transforms(
            "pydantic", "from pydantic import BaseSettings\n"
        )
        assert result.ok is True
        assert result.changed is True
        assert "pydantic.basesettings_import_move" in result.applied
        assert "from pydantic_settings import BaseSettings" in result.source
        # The old import must be gone.
        assert "from pydantic import BaseSettings" not in result.source
        # R3.4: the result must be valid Python.
        assert rt.validate_python_source(result.source)

    def test_mixed_import_splits_and_keeps_other_names_on_pydantic(self) -> None:
        result = rt.apply_source_transforms(
            "pydantic", "from pydantic import BaseModel, BaseSettings\n"
        )
        assert result.ok is True
        assert "pydantic.basesettings_import_move" in result.applied
        assert "from pydantic_settings import BaseSettings" in result.source
        assert "BaseModel" in result.source
        # R3.4: still valid Python after splitting the import.
        assert rt.validate_python_source(result.source)
        # BaseModel must still come from pydantic, not pydantic_settings.
        tree = ast.parse(result.source)
        base_model_module = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "BaseModel":
                        base_model_module = node.module
        assert base_model_module == "pydantic"

    def test_no_basesettings_leaves_import_untouched(self) -> None:
        result = rt.apply_source_transforms(
            "pydantic", "from pydantic import BaseModel\n"
        )
        assert "pydantic.basesettings_import_move" not in result.applied
        assert "pydantic_settings" not in result.source


class TestPydanticRenames:
    def test_validator_decorator_renamed(self) -> None:
        src = (
            "from pydantic import validator\n"
            "@validator('x')\n"
            "def check(cls, v):\n"
            "    return v\n"
        )
        result = rt.apply_source_transforms("pydantic", src)
        assert result.ok is True
        assert "pydantic.validator_rename" in result.applied
        assert "field_validator" in result.source
        assert "@validator" not in result.source
        assert rt.validate_python_source(result.source)

    def test_dict_call_renamed_to_model_dump(self) -> None:
        src = "def f(user):\n    return user.dict()\n"
        result = rt.apply_source_transforms("pydantic", src)
        assert result.ok is True
        assert "pydantic.dict_to_model_dump" in result.applied
        assert "user.model_dump()" in result.source
        assert rt.validate_python_source(result.source)

    def test_dict_call_with_arguments_left_untouched(self) -> None:
        # Argument-bearing .dict(...) is not mechanically safe; leave it alone.
        src = "def f(user):\n    return user.dict(exclude={'x'})\n"
        result = rt.apply_source_transforms("pydantic", src)
        assert "pydantic.dict_to_model_dump" not in result.applied
        assert "user.dict(exclude=" in result.source

    def test_builtin_dict_constructor_is_not_renamed(self) -> None:
        # A bare dict() constructor is not an attribute call, so it is safe.
        src = "value = dict()\n"
        result = rt.apply_source_transforms("pydantic", src)
        assert result.changed is False
        assert "model_dump" not in result.source


class TestPydanticFixtureFile:
    """End-to-end transform over the demo repo's real pydantic fixture."""

    FIXTURE = (
        '"""Pydantic 1.x-style models."""\n'
        "\n"
        "from pydantic import BaseModel, validator\n"
        "\n"
        "\n"
        "class User(BaseModel):\n"
        "    id: int\n"
        "    email: str\n"
        "\n"
        "    class Config:\n"
        "        anystr_strip_whitespace = True\n"
        "\n"
        "    @validator('email')\n"
        "    def email_must_contain_at(cls, value):\n"
        "        if '@' not in value:\n"
        "            raise ValueError('email must contain @')\n"
        "        return value\n"
        "\n"
        "\n"
        "def user_summary(user):\n"
        "    return user.dict()\n"
    )

    def test_autofixes_applied_and_config_flagged(self) -> None:
        result = rt.apply_source_transforms("pydantic", self.FIXTURE)

        assert result.ok is True
        assert result.changed is True
        # The two mechanical renames present in the fixture are applied.
        assert "pydantic.validator_rename" in result.applied
        assert "pydantic.dict_to_model_dump" in result.applied
        assert "field_validator" in result.source
        assert "user.model_dump()" in result.source

        # R3.3: the class-based Config is NOT transformed; it is flagged, and the
        # code for it is left untouched.
        flagged_ids = {f["pattern_id"] for f in result.flagged}
        assert "pydantic.config_class_to_model_config" in flagged_ids
        assert "class Config:" in result.source
        assert "anystr_strip_whitespace = True" in result.source

        # R3.4: the transformed file remains valid Python.
        assert rt.validate_python_source(result.source)


class TestRequestsNoTransform:
    def test_requests_source_is_never_code_transformed(self) -> None:
        # requests is version-bump-only: source is returned unchanged.
        src = "import requests\n\nrequests.get('https://example.com')\n"
        result = rt.apply_source_transforms("requests", src)
        assert result.ok is True
        assert result.changed is False
        assert result.applied == []
        assert result.source == src


# --- manifest version bump (R3.2) -----------------------------------------


class TestBumpManifestVersion:
    def test_requirements_txt_pin_bumped(self) -> None:
        result = rt.bump_manifest_version("requests==2.31.0\n", "requests", "2.32.3")
        assert result.changed is True
        assert result.old_versions == ["2.31.0"]
        assert result.content == "requests==2.32.3\n"

    def test_pyproject_pin_bumped_preserving_quotes(self) -> None:
        manifest = 'dependencies = [\n    "pydantic==1.10.13",\n]\n'
        result = rt.bump_manifest_version(manifest, "pydantic", "2.9.2")
        assert result.changed is True
        assert result.old_versions == ["1.10.13"]
        assert '"pydantic==2.9.2"' in result.content

    def test_only_the_named_package_is_bumped(self) -> None:
        manifest = 'dependencies = [\n    "requests==2.31.0",\n    "pydantic==1.10.13",\n]\n'
        result = rt.bump_manifest_version(manifest, "pydantic", "2.9.2")
        assert '"requests==2.31.0"' in result.content
        assert '"pydantic==2.9.2"' in result.content

    def test_name_matching_is_separator_insensitive(self) -> None:
        result = rt.bump_manifest_version(
            "Flask_SQLAlchemy==2.5.1\n", "flask-sqlalchemy", "3.1.1"
        )
        assert result.changed is True
        assert "==3.1.1" in result.content

    def test_extras_group_preserved(self) -> None:
        result = rt.bump_manifest_version(
            "requests[security]==2.31.0\n", "requests", "2.32.3"
        )
        assert result.changed is True
        assert "requests[security]==2.32.3" in result.content

    def test_no_matching_pin_reports_unchanged(self) -> None:
        result = rt.bump_manifest_version("flask==1.0.0\n", "requests", "2.32.3")
        assert result.changed is False
        assert result.old_versions == []
        assert result.content == "flask==1.0.0\n"


# --- invalid-AST rejection & safe fallback (R3.4, R3.5) -------------------


class _BreakingTransformer(rt.cst.CSTTransformer):
    """A deliberately destructive transformer that yields invalid Python.

    It rewrites every ``Name`` to the reserved keyword ``def``, which parses in
    the CST but produces source that the stdlib parser rejects — exactly the
    condition R3.4/R3.5 must catch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.applied = True

    def leave_Name(self, original_node, updated_node):  # noqa: N802, ANN001
        return updated_node.with_changes(value="def")


class TestInvalidAstRejection:
    def test_transformed_invalid_source_is_rejected_and_original_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a pattern whose transform produces invalid Python.
        bad_pattern = rt.TransformPattern(
            pattern_id="test.breaking",
            description="produces invalid code on purpose",
            auto_fixable=True,
            build=_BreakingTransformer,
        )
        monkeypatch.setitem(rt.PATTERN_REGISTRY, "brokenpkg", [bad_pattern])

        original = "value = 1\n"
        result = rt.apply_source_transforms("brokenpkg", original)

        # R3.4/R3.5: invalid transform is rejected, original is preserved, and
        # the result signals failure so the caller can fall back.
        assert result.ok is False
        assert result.changed is False
        assert result.source == original
        assert result.applied == []
        assert result.error is not None

    def test_unparseable_input_source_is_rejected(self) -> None:
        result = rt.apply_source_transforms("pydantic", "def broken(:\n")
        assert result.ok is False
        assert result.source == "def broken(:\n"


# --- apply_migration executor boundary (R3.2, R3.3, R3.5) -----------------


class TestApplyMigration:
    def test_requests_version_bump_only(self) -> None:
        plan = {
            "package": "requests",
            "current": "2.31.0",
            "target": "2.32.3",
            "strategy": rt.STRATEGY_AUTO_FIX,
        }
        manifest = ("requirements.txt", "requests==2.31.0\n")
        result = rt.apply_migration(plan, manifest=manifest)

        assert result["strategy"] == rt.STRATEGY_AUTO_FIX
        assert result["changes"]["requirements.txt"] == "requests==2.32.3\n"
        assert result["applied"] == []
        assert result["flagged"] == []
        assert "2.32.3" in result["diff"]

    def test_pydantic_autofix_plus_manifest_and_flag(self) -> None:
        plan = {
            "package": "pydantic",
            "current": "1.10.13",
            "target": "2.9.2",
            "strategy": rt.STRATEGY_AUTO_FIX,
        }
        manifest = ("pyproject.toml", 'dependencies = ["pydantic==1.10.13"]\n')
        source_files = {
            "models.py": (
                "from pydantic import BaseModel, validator\n"
                "class User(BaseModel):\n"
                "    class Config:\n"
                "        anystr_strip_whitespace = True\n"
                "    @validator('x')\n"
                "    def check(cls, v):\n"
                "        return v\n"
            ),
        }
        result = rt.apply_migration(
            plan, source_files=source_files, manifest=manifest
        )

        # Manifest bumped (R3.2).
        assert '"pydantic==2.9.2"' in result["changes"]["pyproject.toml"]
        # Auto-fix applied to the source (R3.1).
        assert "field_validator" in result["changes"]["models.py"]
        assert "pydantic.validator_rename" in result["applied"]
        # Config flagged for human review (R3.3), and its code left intact.
        flagged_ids = {f["pattern_id"] for f in result["flagged"]}
        assert "pydantic.config_class_to_model_config" in flagged_ids
        # Every transformed file remains valid Python (R3.4).
        assert rt.validate_python_source(result["changes"]["models.py"])

    def test_guided_pr_strategy_bumps_manifest_but_leaves_code(self) -> None:
        # A guided_pr plan must not auto-transform code (R5.2): version bump +
        # flagged notes only.
        plan = {
            "package": "pydantic",
            "current": "1.10.13",
            "target": "2.9.2",
            "strategy": rt.STRATEGY_GUIDED_PR,
            "breaking_changes": ["Config class must become model_config"],
        }
        manifest = ("pyproject.toml", 'dependencies = ["pydantic==1.10.13"]\n')
        source_files = {"models.py": "def f(u):\n    return u.dict()\n"}
        result = rt.apply_migration(
            plan, source_files=source_files, manifest=manifest
        )

        assert result["strategy"] == rt.STRATEGY_GUIDED_PR
        # Manifest still bumped.
        assert '"pydantic==2.9.2"' in result["changes"]["pyproject.toml"]
        # Code NOT transformed.
        assert "models.py" not in result["changes"]
        assert result["applied"] == []
        # Planner-provided breaking change carried through as flagged.
        descriptions = [f["description"] for f in result["flagged"]]
        assert any("model_config" in d for d in descriptions)

    def test_unsafe_transform_falls_back_to_guided_pr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # R3.5: when a transform would produce invalid code, drop the code
        # change, keep the safe manifest bump, and resolve to guided_pr.
        bad_pattern = rt.TransformPattern(
            pattern_id="test.breaking",
            description="produces invalid code on purpose",
            auto_fixable=True,
            build=_BreakingTransformer,
        )
        monkeypatch.setitem(rt.PATTERN_REGISTRY, "brokenpkg", [bad_pattern])

        plan = {
            "package": "brokenpkg",
            "current": "1.0.0",
            "target": "2.0.0",
            "strategy": rt.STRATEGY_AUTO_FIX,
        }
        manifest = ("requirements.txt", "brokenpkg==1.0.0\n")
        source_files = {"code.py": "value = 1\n"}
        result = rt.apply_migration(
            plan, source_files=source_files, manifest=manifest
        )

        # Fell back to guided_pr (never committed broken code).
        assert result["strategy"] == rt.STRATEGY_GUIDED_PR
        # The broken source is NOT in changes.
        assert "code.py" not in result["changes"]
        # The safe manifest bump survived.
        assert result["changes"]["requirements.txt"] == "brokenpkg==2.0.0\n"
        assert result["applied"] == []

    def test_missing_package_key_raises(self) -> None:
        with pytest.raises(KeyError):
            rt.apply_migration({"target": "2.0.0"})
