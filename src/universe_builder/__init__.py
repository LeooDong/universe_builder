"""A domain-agnostic pipeline for generating a structured, validated JSON corpus with LLMs."""

#: The one source of the release number: `pyproject.toml` reads it
#: (`[tool.setuptools.dynamic]`), and the journal records it on every run.
#: Said `0.1.0` through tag v0.1.10; bump it with the tag.
__version__ = "0.7.0"
