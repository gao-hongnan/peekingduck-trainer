from pathlib import Path
import inspect
from typing import _SpecialForm, get_args, get_origin, Any, get_type_hints
from functools import wraps
import logging
from dataclasses import _HAS_DEFAULT_FACTORY_CLASS


def _find_type_origin(type_hint):
    if isinstance(type_hint, _SpecialForm):
        # case of typing.Any, typing.ClassVar, typing.Final, typing.Literal,
        # typing.NoReturn, typing.Optional, or typing.Union without parameters
        return

    actual_type = get_origin(type_hint) or type_hint  # requires Python 3.8
    if isinstance(actual_type, _SpecialForm):
        # case of typing.Union[…] or typing.ClassVar[…] or …
        for origins in map(_find_type_origin, get_args(type_hint)):
            yield from origins
    else:
        yield actual_type


def _check_types(parameters, hints):
    for name, value in parameters.items():
        if isinstance(value, _HAS_DEFAULT_FACTORY_CLASS):
            return  # FIXME: how to handle default_factory? uncomment to see error.
        type_hint = hints.get(name, Any)
        actual_types = tuple(_find_type_origin(type_hint))
        if actual_types and not isinstance(value, actual_types):
            raise TypeError(
                f"Expected type '{type_hint}' for argument '{name}'"
                f" but received type '{type(value)}' instead"
            )


def enforce_types(wrapped):
    """Decorator to enforce type hints at runtime.

    Note:
        - This decorator is not compatible with default_factory.
        - See https://github.com/tamuhey/dataclass_utils for more details to implement.

    References:
        - https://github.com/matchawine/python-enforce-typing
        - https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses
    """

    def decorate(func):
        hints = get_type_hints(func)
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = dict(zip(signature.parameters, bound.args))
            parameters.update(bound.kwargs)
            _check_types(parameters, hints)
            return func(*args, **kwargs)

        return wrapper

    if inspect.isclass(wrapped):
        wrapped.__init__ = decorate(wrapped.__init__)
        return wrapped
    return decorate(wrapped)


class Validator:  # pylint: disable=too-few-public-methods
    """Validator class to be inherited by dataclasses.

    Reference: https://gist.github.com/rochacbruno/978405e4839142e409f8402eece505e8
    """

    # FIXME: currently do not work well because it uses __post_init__
    # and some dataclass configs also has __post_init__, which overwrites this.
    # either change name or use a different way to run validation.
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        """Run validation methods if declared.
        The validation method can be a simple check
        that raises ValueError or a transformation to
        the field value.
        The validation is performed by calling a function named:
            `validate_<field_name>(self, value, field) -> field.type`
        """
        if not hasattr(self, "__dataclass_fields__"):
            raise ValueError(
                f"{self.__class__.__name__} must be a dataclass "
                f"with type annotations."
            )
        for name, field in self.__dataclass_fields__.items():
            validator_name = f"validate_{name}"
            if method := getattr(self, f"validate_{name}", None):
                print(f"Running validation method {validator_name}.")
                # self.logger.info(f"Running validation method {validator_name}.")
                value = method(getattr(self, name), field=field)
                # call the method with the field value and the field
                setattr(self, name, value)
