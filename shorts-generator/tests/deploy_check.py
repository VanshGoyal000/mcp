#!/usr/bin/env python
"""
Simple script to verify Modal deployment status and connectivity.
Use this to check if your Modal functions are properly deployed and accessible.
"""

import os
import sys
from pathlib import Path
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

def check_modal_installation():
    """Check if Modal is properly installed"""
    print("Checking Modal installation...")
    try:
        import modal
        print(f"✓ Modal package is installed (version: {modal.__version__})")
        return True
    except ImportError:
        print("✗ Modal package is not installed")
        print("  To install Modal: pip install modal")
        return False
    except Exception as e:
        print(f"✗ Error importing Modal: {e}")
        return False

def check_modal_deployment():
    """Check if Modal app is deployed and accessible"""
    print("\nChecking Modal deployment status...")
    try:
        import modal
        
        # Try to access the app by name
        print("Attempting to access Modal app 'shorts-generator'...")
        app = modal.App.from_name("shorts-generator")
        
        # List available functions
        functions = [f for f in dir(app) if not f.startswith('_') and callable(getattr(app, f))]
        
        if functions:
            print(f"✓ Modal app 'shorts-generator' is deployed with {len(functions)} functions:")
            for func in functions:
                print(f"  - {func}")
            return True
        else:
            print("✗ Modal app 'shorts-generator' is deployed but has no functions")
            print("  Try redeploying with: python ../modal_deploy.py")
            return False
            
    except AttributeError as e:
        print(f"✗ Modal API error: {e}")
        print("  Your version of Modal might be using a different API.")
        print("  Update modal_config.py to use the correct API functions.")
        return False
    except Exception as e:
        print(f"✗ Could not access Modal deployment: {e}")
        print("  Make sure you've deployed the app with: python ../modal_deploy.py")
        return False

def check_modal_config():
    """Check if modal_config.py is properly set up"""
    print("\nChecking modal_config.py...")
    try:
        from modal_config import app, get_secrets, get_base_image, get_modal_functions
        
        # Check if app has any functions
        functions = [f for f in dir(app) if not f.startswith('_') and callable(getattr(app, f))]
        
        if functions:
            print(f"✓ modal_config.py is properly set up with {len(functions)} functions")
            for func in functions:
                print(f"  - {func}")
        else:
            print("✗ modal_config.py is loaded but app has no functions")
            print("  Functions might not be properly connected.")
            print("  Check get_modal_functions() implementation in modal_config.py")
            
        return True
        
    except ImportError:
        print("✗ Could not import modal_config module")
        print("  Make sure modal_config.py exists in the project root")
        return False
    except Exception as e:
        print(f"✗ Error in modal_config.py: {e}")
        print("  Check for syntax errors or missing requirements")
        traceback.print_exc()
        return False

def main():
    """Run all deployment checks"""
    print("======= Modal Deployment Status Checker =======\n")
    
    modal_installed = check_modal_installation()
    if not modal_installed:
        print("\nERROR: Modal is not properly installed.")
        print("Please install Modal before continuing.")
        return
        
    config_ok = check_modal_config()
    if not config_ok:
        print("\nERROR: Modal configuration is not properly set up.")
        print("Please fix the configuration before continuing.")
    
    deployed = check_modal_deployment()
    if not deployed:
        print("\nERROR: Modal app is not properly deployed or accessible.")
        print("Deploy the app first with: python ../modal_deploy.py")
    
    print("\n============ Summary ============")
    if modal_installed and config_ok and deployed:
        print("✓ All checks passed! Modal is properly set up and deployed.")
        print("  You can now run the test script.")
    else:
        print("✗ Some checks failed. Please fix the issues before running tests.")

if __name__ == "__main__":
    main()
