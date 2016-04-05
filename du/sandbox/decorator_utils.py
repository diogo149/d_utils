import functools
import contextlib


def to_decorator(wrapped_func):
    """
    Encapsulates the decorator logic for most common use cases.
    Expects a wrapped function with compatible type signature to:
        wrapped_func(func, args, kwargs, *outer_args, **outer_kwargs)
    Example:
    @to_decorator
    def foo(func, args, kwargs):
        print(func)
        return func(*args, **kwargs)
    @foo()
    def bar():
        print(42)
    """
    @functools.wraps(wrapped_func)
    def arg_wrapper(*outer_args, **outer_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return wrapped_func(func,
                                    args,
                                    kwargs,
                                    *outer_args,
                                    **outer_kwargs)
            return wrapped
        return decorator
    return arg_wrapper


def g_decorator(generator_expr):
    """
    Converts generator expression into a decorator
    Takes in a generator expression, such as one accepted by
    contextlib.contextmanager, converts it to a context manager,
    and returns a decorator equivalent to being within that
    context manager.
    TODO do something with yielded value
    Example:
    @g_decorator
    def foo():
        print("Hello")
        yield
        print("World")
    @foo()
    def bar():
        print("Something")
    """
    cm = contextlib.contextmanager(generator_expr)

    @to_decorator
    def wrapped_func(func, args, kwargs, *outer_args, **outer_kwargs):
        with cm(*outer_args, **outer_kwargs) as yielded:
            return func(*args, **kwargs)

    return wrapped_func
