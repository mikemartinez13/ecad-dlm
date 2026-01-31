from ecad.image_generators.dream_text_generator import (
    DreamTextGenerator,
)
from ecad.image_generators.text_generator import (
    TextGenerator,
)

from ecad.types import TextGeneratorConfig


class TextGeneratorRegistry:
    registry: dict[str, type[TextGenerator]] = {
        "DreamTextGenerator": DreamTextGenerator,
    }

    @classmethod
    def get(
        cls, name: str, default_name: str | None = None
    ) -> type[TextGenerator] | None:
        """Get the text generator class by name, or the default text generator class if not found.

        Args:
            name: the name the text generator was registered with.
            default_name: the default text generator name to return if the named text generator is not found.
        Returns:
            The text generator class.
        """
        text_generator = cls.registry.get(name, None)
        if text_generator is None and default_name is not None:
            text_generator = cls.registry.get(default_name, None)
        return text_generator


def get_text_generator_type_from_config(
    config: TextGeneratorConfig, default_name: str = "DreamTextGenerator"
) -> type[TextGenerator]:
    """Get the text generator class by name.

    Args:
        config: the text generator configuration.
        default_name: the default text generator name.

    Returns:
        The text generator class.
    """
    if "text_generator" not in config:
        text_gen = None
    else:
        text_gen = TextGeneratorRegistry.get(
            config["text_generator"], default_name
        )

    if text_gen is None:
        raise ValueError(f"Text generator not found in config: {config}.")

    return text_gen

def get_text_generator_type(
    name: str, default_name: str = "DreamTextGenerator"
) -> type[TextGenerator]:
    """Get the text generator class by name.

    Args:
        name: the name of the text generator.
        default_name: the default text generator name.

    Returns:
        The text generator class.
    """
    text_gen = TextGeneratorRegistry.get(name, default_name)
    if text_gen is None:
        raise ValueError(f"Text generator not found: {name}.")

    return text_gen