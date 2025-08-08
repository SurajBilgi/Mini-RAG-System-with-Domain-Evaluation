#!/usr/bin/env python3
"""
Installation script for Mini RAG System
Helps resolve common dependency conflicts and setup issues
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a command and handle errors gracefully"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=check
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def install_dependencies():
    """Install dependencies with proper error handling"""
    print("\n📦 Installing dependencies...")

    # Upgrade pip first
    print("Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Install dependencies in specific order to avoid conflicts
    critical_packages = [
        "wheel",
        "setuptools",
        "numpy==1.24.3",
        "pandas==2.1.4",
    ]

    for package in critical_packages:
        print(f"Installing {package}...")
        result = run_command(f"{sys.executable} -m pip install {package}")
        if result and result.returncode != 0:
            print(f"⚠️ Warning: Failed to install {package}")

    # Install remaining packages
    print("Installing remaining dependencies...")
    result = run_command(f"{sys.executable} -m pip install -r requirements.txt")

    if result and result.returncode == 0:
        print("✅ Dependencies installed successfully!")
        return True
    else:
        print("❌ Some dependencies failed to install")
        return False


def verify_installation():
    """Verify that key packages are properly installed"""
    print("\n🔍 Verifying installation...")

    test_imports = [
        ("streamlit", "Streamlit web framework"),
        ("langchain", "LangChain framework"),
        ("openai", "OpenAI API client"),
        ("sentence_transformers", "Sentence transformers"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]

    failed_imports = []

    for package, description in test_imports:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n⚠️ Failed imports: {', '.join(failed_imports)}")
        print("Try running: pip install --force-reinstall <package_name>")
        return False
    else:
        print("\n🎉 All packages verified successfully!")
        return True


def create_sample_env():
    """Create a sample .env file for environment variables"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Set default model
DEFAULT_MODEL=gpt-3.5-turbo

# Optional: Set embedding model
DEFAULT_EMBEDDING_MODEL=text-embedding-ada-002
"""

    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("📝 Created sample .env file")
        print("Please edit .env and add your OpenAI API key")
    else:
        print("📝 .env file already exists")


def main():
    """Main installation process"""
    print("🚀 Mini RAG System Installation")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        print("Make sure you're in the correct directory")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed!")
        print("\nTroubleshooting tips:")
        print("1. Try creating a new virtual environment:")
        print("   python -m venv rag_env")
        print(
            "   source rag_env/bin/activate  # On Windows: rag_env\\Scripts\\activate"
        )
        print("   python install.py")
        print("\n2. If using conda:")
        print("   conda create -n rag_env python=3.10")
        print("   conda activate rag_env")
        print("   python install.py")
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("\n⚠️ Installation completed with some issues")
        print("The application may still work, but some features might be limited")

    # Create sample environment file
    create_sample_env()

    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run the application: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")


if __name__ == "__main__":
    main()
