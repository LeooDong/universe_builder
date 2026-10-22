import logging
import json
from pathlib import Path

from universe_builder import settings

from universe_builder import store

settings.load_environment()
log = logging.getLogger(__name__)


class LLMCall:
    def __init__(
        self,
        atom_type,
        atom_class,
        model_version,
        prompt_folder,
        prompt_subject,
        prompt_start_text,
        base_system_text,
        example_json_file,
        target_path=None,
        supplements_target=False,
        max_tokens=None,
        operation=None,
        context=None,
    ):

        # TODO: clean up the code later

        log.warning(f"            INITIALISING LLM CALL FOR {prompt_subject}")
        # Prefer the active domain pack's own examples dir; fall back to the
        # package examples for that prompt folder (history's, and the default).
        # This is what lets a synchronous run of a second domain find its own
        # one-shot examples without every caller threading a folder through.
        from universe_builder.registry import active_pack
        self.atom_location = active_pack().examples_dir or settings.examples_dir(prompt_folder)

        # setting self variables -------------------------------------------------------------------
        self.atom_type = atom_type
        self.atom_class = atom_class

        self.prompt_subject = prompt_subject
        self.model_version = model_version
        self.base_system_text = base_system_text
        #: The output-token budget the operation asked for (`call_budget`);
        #: None is the subclass's own default. `ProviderCall` sets its own.
        self.max_tokens = max_tokens
        #: The operation whose instructions the request carries
        #: (`prompts/operations/<operation>.txt`) and the framing context (the
        #: civilisation), for the call classes that build their request with
        #: `providers.requests.build_request` (`ProviderCall`). None leaves the
        #: class's own default; classes that build their own prompt ignore both.
        self.prompt_operation = operation
        self.context = context
        self.schema_json = self.atom_class.model_json_schema()

        # setting optional self variables ----------------------------------------------------------
        self.prompt_start_text = prompt_start_text if prompt_start_text is not None else None
        self.example_json_file = example_json_file if example_json_file is not None else None

        log.debug("Loading filename for saved file")
        self.response_path = store.path_for(
            self.atom_type, None, self.prompt_subject, self.model_version
        )
        #: Whether a validation failure writes an `_ERROR` sentinel. True for
        #: calls that create their own file; turned off by
        #: `_inherit_target_identity` for calls that work *on* an existing one.
        self.writes_error_sentinel = True
        if target_path is not None:
            self._inherit_target_identity(target_path, supplements_target)
        # Kept as a string for the subclasses and tests that still read it.
        self.response_file_location = str(self.response_path)

        self.validate_example_json()

        log.debug(f"Created LLM call for {self.response_path.name}")

    def _inherit_target_identity(self, target_path, supplements_target: bool) -> None:
        """Work under the target's identity, never a fresh one minted from a slug.

        An augment call's `prompt_subject` is read off the target's filename,
        and slugs are lossy — `Boutros Boutros-Ghali` slugs to
        `boutros_boutros_ghali`, and nothing recovers the original — so hashing
        the slug minted a *second* identity for work on an existing file: a
        failed fix wrote its `_ERROR` under a name nothing owned (a phantom in
        the fix queue), and an append's id never matched the city it
        supplements.

        The caller states the intent; it is not inferred. A first version
        keyed on whether the target's filename shared the call's atom prefix,
        which misclassified *expand* — its calls carry `atom_type=expand_city`
        against a `city_…` target, so they fell into the supplement branch and
        kept minting orphan sentinels, the exact defect this exists to close.

        Not supplementing (fix, verify, expand — the call is *about* the
        target file): `response_path` becomes the target's own path and no
        `_ERROR` sentinel is written on failure. For a fix, the sentinel
        already there is the input being repaired and must survive (B8); for a
        verify or expand, the target is a valid atom and a sentinel beside it
        would queue a "repair" that overwrites good data with output derived
        from a failed attempt. The run journal records the failed payload
        either way, so no evidence is lost.

        Supplementing (append): the call still creates its own file — sentinel
        behaviour unchanged — but inherits the target's *id*, which is what
        lets `merge_helper` pair the two on identity whatever model wrote
        either half.
        """
        target = store.target_of_error(Path(target_path))
        if not supplements_target:
            self.response_path = target
            self.writes_error_sentinel = False
            return
        try:
            parsed = store.AtomName.decompose_filename(target.name)
        except ValueError:
            return  # not a pipeline filename; keep the minted identity
        if store.is_hash_id(parsed.padded_id):
            self.response_path = store.path_for(
                self.atom_type, parsed.padded_id, self.prompt_subject, self.model_version
            )

    def validate_example_json(self) -> None:
        """Load and validate the configured one-shot or few-shot examples.

        `example_json_file` in an AtomSpec may be a single filename or a
        list of them. Both forms are normalised here so every consumer sees the
        same shape: `example_json_files` / `example_jsons` hold all of them.
        (The singular `example_json` alias, and the warning that a non-provider
        class used only the first, went in v0.3.0.)
        """
        configured = self.example_json_file
        if not configured:
            # we've unset the example_json_file to try this as a dry run
            self.example_json_files = []
            self.example_jsons = []
            log.debug("Running this call as a one shot call")
            return

        filenames = [configured] if isinstance(configured, str) else list(configured)
        self.example_json_files = filenames
        self.example_jsons = []

        for filename in filenames:
            with open(f"{self.atom_location}/{filename}", "r") as f:
                log.debug(f"Opened example json file {filename}")
                example = json.load(f)
            log.debug(f"Attempting to validate {filename} with defined atom")
            try:
                self.atom_class.model_validate(example)
                log.info(f"Success validating {filename}!")
            except Exception as e:
                raise ValueError(f"Example JSON {filename} has not been validated: {e}")
            self.example_jsons.append(example)

    def log_request(self):
        log.debug(f"""{self.example_text}""")
        log.debug(f"""{self.example_jsons}""")
        log.debug(f"""{self.system_text}""")
        log.debug(f"""{self.prompt_text}""")

    def call_llm_api():
        raise NotImplementedError("Subclass must implement call_llm_api method")

    #: Set by a subclass when the transport already judged the response unusable.
    #: `None` means "no verdict", not "fine" — see `parse_response_to_pydantic`.
    provider_error = None
    #: Token counts from the provider, set by subclasses that measure them
    #: (`ProviderCall`). `None` means "this transport reported nothing", which
    #: the journal records honestly as `usage_available: false` rather than as
    #: zero tokens — a call that cost something unknown is not a free call.
    usage = None

    def parse_response_to_pydantic(self):
        if self.provider_error:
            # The provider already refused this, and re-validating would overturn
            # it. That is not hypothetical: an incomplete stream cut after the
            # last brace yields JSON that parses perfectly, so
            # `model_validate` succeeded and the pipeline wrote a truncated
            # atom and reported `outcome: ok` — while the provider had logged
            # "treating it as incomplete" one frame earlier.
            #
            # Validity of the *text* is not the question the provider answered.
            # It answered whether the text is the whole response.
            if getattr(self, "writes_error_sentinel", True):
                store.write_error(self.response_path, self.llm_response_json)
            self.atom_instance = store.get_empty_model()
            log.error("%s: %s", self.prompt_subject, self.provider_error)
            return

        try:
            log.debug(f"LLM response before validation was: {self.llm_response_json}")
            self.atom_instance = self.atom_class.model_validate(json.loads(self.llm_response_json))

            log.info(f"Success validating {self.prompt_subject}!")
        except Exception as e:
            if getattr(self, "writes_error_sentinel", True):
                store.write_error(self.response_path, self.llm_response_json)
            self.atom_instance = store.get_empty_model()

            # Not "the example": the *response* is what failed to validate. The
            # old wording sent readers to inspect a bundled example file that was
            # never the problem.
            log.error("Response for %s did not validate: %s", self.prompt_subject, e)
            # raise ValueError(f"Example JSON has not been validated: {e}")

    def write_response_to_file(self):
        self.parse_response_to_pydantic()
        store.write_atom(self.response_path, self.atom_instance)

    def get_valid_response(self) -> object:
        self.parse_response_to_pydantic()
        return self.atom_instance
