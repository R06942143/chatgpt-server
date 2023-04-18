from pydantic import validate_arguments


# Should always use the singleton
class BaseStore(object):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(
                    cls,
                    attr,
                    validate_arguments(
                        getattr(cls, attr), config=dict(arbitrary_types_allowed=True)
                    ),
                )
