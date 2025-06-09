# Simplify imports - we'll handle the path in run.py
__all__ = ["create_app"]

# Only attempt the import when this module is actually being used
# This prevents circular import errors
def get_create_app():
    from ui.app import create_app
    return create_app
