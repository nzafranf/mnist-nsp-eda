#!/usr/bin/env python3
"""
Setup script to install PDF conversion dependencies.

This script installs the necessary packages for markdown to PDF conversion.
"""

import subprocess
import sys


def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def main():
    packages = [
        "markdown2",      # Markdown parser
        "weasyprint",     # HTML to PDF converter (lightweight)
    ]

    print("Setting up PDF converter dependencies...")
    print("This may take a few minutes on first install.\n")

    try:
        for package in packages:
            install_package(package)

        print("\n✓ All dependencies installed successfully!")
        print("\nYou can now run: python convert_md_to_pdf.py")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to install packages: {e}")
        print("\nAlternative installation (if the above fails):")
        print("  pip install markdown2 weasyprint")
        sys.exit(1)


if __name__ == "__main__":
    main()
