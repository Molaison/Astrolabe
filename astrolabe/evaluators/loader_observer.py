"""Count successful backend loader calls without changing their inputs/results."""
from functools import wraps


class LoaderObserver:
    def __init__(self, owner, attribute):
        self.owner = owner
        self.attribute = attribute
        self.original = getattr(owner, attribute)
        self.count = 0
        self.failures = 0
        @wraps(self.original)
        def observed(*args, **kwargs):
            try:
                result = self.original(*args, **kwargs)
            except BaseException:
                self.failures += 1
                raise
            self.count += 1
            return result
        self.wrapped = observed
        setattr(owner, attribute, observed)

    def close(self):
        if getattr(self.owner, self.attribute) is self.wrapped:
            setattr(self.owner, self.attribute, self.original)
