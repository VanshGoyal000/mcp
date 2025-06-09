import os
import asyncio
import sys
import socket
import logging

logger = logging.getLogger(__name__)

def apply_windows_asyncio_fixes():
    """
    Applies fixes for the Windows asyncio ConnectionResetError that floods the console.
    This is a harmless error that occurs when connections are closed, but it produces 
    distracting error messages.
    """
    if os.name != 'nt':  # Only apply these fixes on Windows
        return

    # Create custom exception handler to ignore ConnectionResetErrors
    def custom_exception_handler(loop, context):
        exception = context.get('exception')
        if isinstance(exception, ConnectionResetError) and "[WinError 10054]" in str(exception):
            # Just log at debug level without full traceback
            logger.debug("Suppressed ConnectionResetError - normal on Windows")
            return
        # For all other exceptions, use the default exception handler
        loop.default_exception_handler(context)
    
    # Apply the custom exception handler
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(custom_exception_handler)
        logger.info("Applied Windows-specific asyncio fixes")
    except Exception as e:
        logger.warning(f"Failed to apply Windows asyncio fixes: {e}")
        
    # Apply additional fixes for Windows proactor issues
    if sys.version_info >= (3, 8):
        # Use selector event loop policy instead of proactor on Windows
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Set Windows Selector event loop policy")
        except Exception:
            pass
