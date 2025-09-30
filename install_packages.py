#!/usr/bin/env python3
"""
Comprehensive Python Package Installation Script
Installs common data science, optimization, and analysis packages
Robust error handling and logging for production environments
Supports virtual environment creation and management

# Quick start - new project with venv
python install_packages.py --create-venv

# Just optimization packages for your CVaR work
python install_packages.py --create-venv --categories core,optimization,viz,financial

# Install everything in current conda environment
python install_packages.py

# Use existing venv but install only ML packages
python install_packages.py --venv-path ./my_env --categories core,ml
"""

import subprocess
import sys
import logging
import time
import os
import argparse
import venv
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import importlib
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('package_installation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VirtualEnvManager:
    """Manages virtual environment creation and activation"""
    
    def __init__(self, env_path: str = None):
        self.env_path = Path(env_path) if env_path else Path("./datascience_env")
        self.original_path = sys.path.copy()
        self.original_executable = sys.executable
        
    def create_venv(self, force: bool = False) -> bool:
        """Create a new virtual environment"""
        try:
            if self.env_path.exists():
                if force:
                    logger.info(f"🗑️  Removing existing venv at {self.env_path}")
                    shutil.rmtree(self.env_path)
                else:
                    logger.info(f"✅ Virtual environment already exists at {self.env_path}")
                    return True
            
            logger.info(f"🏗️  Creating virtual environment at {self.env_path}")
            venv.create(self.env_path, with_pip=True)
            logger.info("✅ Virtual environment created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get path to Python executable in virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.env_path / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.env_path / "bin" / "python")
    
    def get_activation_command(self) -> str:
        """Get the command to activate the virtual environment"""
        if os.name == 'nt':  # Windows
            return f"{self.env_path}\\Scripts\\activate"
        else:  # Unix/Linux/macOS
            return f"source {self.env_path}/bin/activate"
    
    def is_venv_active(self) -> bool:
        """Check if we're running in the virtual environment"""
        return str(self.env_path) in sys.executable


class PackageInstaller:
    """Robust package installer with error handling and verification"""
    
    def __init__(self, timeout: int = 300, python_executable: str = None):
        self.timeout = timeout
        self.python_executable = python_executable or sys.executable
        self.failed_packages = []
        self.success_packages = []
        self.skipped_packages = []
        
    def check_package_installed(self, package_name: str, import_name: Optional[str] = None) -> bool:
        """Check if package is already installed"""
        test_name = import_name or package_name
        try:
            importlib.import_module(test_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package: str, import_name: Optional[str] = None, 
                       extra_args: List[str] = None, conda_name: Optional[str] = None) -> bool:
        """Install a single package with robust error handling"""
        
        # Check if already installed
        if self.check_package_installed(package, import_name):
            logger.info(f"✓ {package} already installed, skipping")
            self.skipped_packages.append(package)
            return True
        
        extra_args = extra_args or []
        
        # Try conda first if conda_name provided and conda is available
        if conda_name and self._has_conda():
            if self._install_with_conda(conda_name, package):
                return True
        
        # Fall back to pip installation
        return self._install_with_pip(package, extra_args)
    
    def _has_conda(self) -> bool:
        """Check if conda is available"""
        try:
            subprocess.run(['conda', '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _install_with_conda(self, conda_name: str, package: str) -> bool:
        """Try installing with conda"""
        try:
            logger.info(f"🔄 Installing {package} via conda...")
            result = subprocess.run(
                ['conda', 'install', '-y', conda_name],
                capture_output=True, text=True, timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {package} via conda")
                self.success_packages.append(package)
                return True
            else:
                logger.warning(f"⚠️  Conda install failed for {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Conda installation timeout for {package}")
            return False
        except Exception as e:
            logger.error(f"❌ Conda installation error for {package}: {e}")
            return False
    
    def _install_with_pip(self, package: str, extra_args: List[str]) -> bool:
        """Install with pip using specified Python executable"""
        try:
            logger.info(f"🔄 Installing {package} via pip...")
            cmd = [self.python_executable, '-m', 'pip', 'install'] + extra_args + [package]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {package}")
                self.success_packages.append(package)
                return True
            else:
                logger.error(f"❌ Failed to install {package}: {result.stderr}")
                self.failed_packages.append((package, result.stderr))
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Installation timeout for {package}")
            self.failed_packages.append((package, "Installation timeout"))
            return False
        except Exception as e:
            logger.error(f"❌ Installation error for {package}: {e}")
            self.failed_packages.append((package, str(e)))
            return False

    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version using specified Python executable"""
        try:
            logger.info("🔄 Upgrading pip...")
            result = subprocess.run(
                [self.python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✓ Successfully upgraded pip")
                return True
            else:
                logger.warning(f"⚠️  Pip upgrade failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Pip upgrade error: {e}")
            return False

    def install_packages_batch(self, packages: Dict[str, Dict]) -> Dict[str, List]:
        """Install multiple packages with batch processing"""
        
        logger.info("🚀 Starting comprehensive package installation...")
        logger.info(f"📦 Total packages to install: {len(packages)}")
        
        # Upgrade pip first
        self.upgrade_pip()
        
        # Install packages
        for package_name, config in packages.items():
            logger.info(f"\n--- Installing {package_name} ---")
            
            try:
                self.install_package(
                    package=config.get('pip_name', package_name),
                    import_name=config.get('import_name'),
                    extra_args=config.get('extra_args', []),
                    conda_name=config.get('conda_name')
                )
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Unexpected error installing {package_name}: {e}")
                self.failed_packages.append((package_name, str(e)))
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, List]:
        """Generate installation report"""
        
        logger.info("\n" + "="*60)
        logger.info("📊 INSTALLATION REPORT")
        logger.info("="*60)
        
        logger.info(f"✅ Successfully installed: {len(self.success_packages)}")
        for pkg in self.success_packages:
            logger.info(f"   • {pkg}")
        
        logger.info(f"⏭️  Already installed (skipped): {len(self.skipped_packages)}")
        for pkg in self.skipped_packages:
            logger.info(f"   • {pkg}")
        
        if self.failed_packages:
            logger.error(f"❌ Failed installations: {len(self.failed_packages)}")
            for pkg, error in self.failed_packages:
                logger.error(f"   • {pkg}: {error}")
        
        total_attempted = len(self.success_packages) + len(self.failed_packages)
        success_rate = (len(self.success_packages) / total_attempted * 100) if total_attempted > 0 else 100
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        return {
            'success': self.success_packages,
            'failed': self.failed_packages,
            'skipped': self.skipped_packages
        }


# Comprehensive package definitions
PACKAGES = {
    # Core Data Science Stack
    'numpy': {
        'pip_name': 'numpy',
        'conda_name': 'numpy',
        'import_name': 'numpy'
    },
    'pandas': {
        'pip_name': 'pandas',
        'conda_name': 'pandas',
        'import_name': 'pandas'
    },
    'scipy': {
        'pip_name': 'scipy',
        'conda_name': 'scipy',
        'import_name': 'scipy'
    },
    
    # Optimization & Operations Research
    'gurobipy': {
        'pip_name': 'gurobipy',
        'conda_name': 'gurobi',
        'import_name': 'gurobipy'
    },
    'cvxpy': {
        'pip_name': 'cvxpy',
        'conda_name': 'cvxpy',
        'import_name': 'cvxpy'
    },
    'pulp': {
        'pip_name': 'pulp',
        'conda_name': 'pulp',
        'import_name': 'pulp'
    },
    'pyomo': {
        'pip_name': 'pyomo',
        'conda_name': 'pyomo',
        'import_name': 'pyomo'
    },
    
    # Machine Learning
    'scikit-learn': {
        'pip_name': 'scikit-learn',
        'conda_name': 'scikit-learn',
        'import_name': 'sklearn'
    },
    'xgboost': {
        'pip_name': 'xgboost',
        'conda_name': 'xgboost',
        'import_name': 'xgboost'
    },
    'lightgbm': {
        'pip_name': 'lightgbm',
        'conda_name': 'lightgbm',
        'import_name': 'lightgbm'
    },
    'catboost': {
        'pip_name': 'catboost',
        'conda_name': 'catboost',
        'import_name': 'catboost'
    },
    
    # Deep Learning (optional - commented out due to size)
    # 'tensorflow': {
    #     'pip_name': 'tensorflow',
    #     'conda_name': 'tensorflow',
    #     'import_name': 'tensorflow'
    # },
    # 'torch': {
    #     'pip_name': 'torch',
    #     'conda_name': 'pytorch',
    #     'import_name': 'torch'
    # },
    
    # Visualization
    'matplotlib': {
        'pip_name': 'matplotlib',
        'conda_name': 'matplotlib',
        'import_name': 'matplotlib'
    },
    'seaborn': {
        'pip_name': 'seaborn',
        'conda_name': 'seaborn',
        'import_name': 'seaborn'
    },
    'plotly': {
        'pip_name': 'plotly',
        'conda_name': 'plotly',
        'import_name': 'plotly'
    },
    'plotnine': {
        'pip_name': 'plotnine',
        'import_name': 'plotnine'
    },
    'bokeh': {
        'pip_name': 'bokeh',
        'conda_name': 'bokeh',
        'import_name': 'bokeh'
    },
    'altair': {
        'pip_name': 'altair',
        'conda_name': 'altair',
        'import_name': 'altair'
    },
    
    # Statistical Analysis
    'statsmodels': {
        'pip_name': 'statsmodels',
        'conda_name': 'statsmodels',
        'import_name': 'statsmodels'
    },
    'pingouin': {
        'pip_name': 'pingouin',
        'import_name': 'pingouin'
    },
    
    # Financial & Quantitative
    'yfinance': {
        'pip_name': 'yfinance',
        'import_name': 'yfinance'
    },
    'quantlib': {
        'pip_name': 'QuantLib',
        'import_name': 'QuantLib'
    },
    'arch': {
        'pip_name': 'arch',
        'import_name': 'arch'
    },
    'pyportfolioopt': {
        'pip_name': 'pyportfolioopt',
        'import_name': 'pypfopt'
    },
    
    # Data Processing & I/O
    'openpyxl': {
        'pip_name': 'openpyxl',
        'conda_name': 'openpyxl',
        'import_name': 'openpyxl'
    },
    'xlsxwriter': {
        'pip_name': 'xlsxwriter',
        'conda_name': 'xlsxwriter',
        'import_name': 'xlsxwriter'
    },
    'requests': {
        'pip_name': 'requests',
        'conda_name': 'requests',
        'import_name': 'requests'
    },
    'beautifulsoup4': {
        'pip_name': 'beautifulsoup4',
        'conda_name': 'beautifulsoup4',
        'import_name': 'bs4'
    },
    'lxml': {
        'pip_name': 'lxml',
        'conda_name': 'lxml',
        'import_name': 'lxml'
    },
    
    # Development & Jupyter
    'jupyter': {
        'pip_name': 'jupyter',
        'conda_name': 'jupyter',
        'import_name': 'jupyter'
    },
    'ipython': {
        'pip_name': 'ipython',
        'conda_name': 'ipython',
        'import_name': 'IPython'
    },
    'tqdm': {
        'pip_name': 'tqdm',
        'conda_name': 'tqdm',
        'import_name': 'tqdm'
    },
    'joblib': {
        'pip_name': 'joblib',
        'conda_name': 'joblib',
        'import_name': 'joblib'
    },
    
    # Utilities
    'python-dotenv': {
        'pip_name': 'python-dotenv',
        'import_name': 'dotenv'
    },
    'click': {
        'pip_name': 'click',
        'conda_name': 'click',
        'import_name': 'click'
    },
    'pyyaml': {
        'pip_name': 'pyyaml',
        'conda_name': 'pyyaml',
        'import_name': 'yaml'
    },
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Python package installer for data science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install in current environment
  python install_packages.py
  
  # Create new venv and install packages
  python install_packages.py --create-venv
  
  # Use existing venv
  python install_packages.py --venv-path ./my_env
  
  # Force recreate venv
  python install_packages.py --create-venv --force
  
  # Install only specific categories
  python install_packages.py --categories core,optimization,viz
        """
    )
    
    # Virtual environment options
    venv_group = parser.add_argument_group('Virtual Environment Options')
    venv_group.add_argument(
        '--create-venv', 
        action='store_true', 
        help='Create a new virtual environment before installing packages'
    )
    venv_group.add_argument(
        '--venv-path', 
        type=str, 
        default='./datascience_env',
        help='Path for virtual environment (default: ./datascience_env)'
    )
    venv_group.add_argument(
        '--force', 
        action='store_true', 
        help='Force recreate virtual environment if it exists'
    )
    
    # Installation options
    install_group = parser.add_argument_group('Installation Options')
    install_group.add_argument(
        '--timeout', 
        type=int, 
        default=300,
        help='Timeout in seconds per package installation (default: 300)'
    )
    install_group.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of package categories to install (e.g., core,ml,viz)'
    )
    install_group.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip packages that are already installed (default behavior)'
    )
    
    return parser.parse_args()


def filter_packages_by_category(packages: Dict, categories: List[str]) -> Dict:
    """Filter packages by category"""
    category_mapping = {
        'core': ['numpy', 'pandas', 'scipy'],
        'optimization': ['gurobipy', 'cvxpy', 'pulp', 'pyomo'],
        'ml': ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'statsmodels'],
        'viz': ['matplotlib', 'seaborn', 'plotly', 'plotnine', 'bokeh', 'altair'],
        'financial': ['yfinance', 'quantlib', 'arch', 'pyportfolioopt'],
        'io': ['openpyxl', 'xlsxwriter', 'requests', 'beautifulsoup4', 'lxml'],
        'dev': ['jupyter', 'ipython', 'tqdm', 'joblib'],
        'utils': ['python-dotenv', 'click', 'pyyaml']
    }
    
    if not categories:
        return packages
    
    selected_packages = set()
    for category in categories:
        if category in category_mapping:
            selected_packages.update(category_mapping[category])
        else:
            logger.warning(f"Unknown category: {category}")
    
    return {k: v for k, v in packages.items() if k in selected_packages}


def main():
    """Main installation function with virtual environment support"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Suppress warnings during installation
    warnings.filterwarnings('ignore')
    
    print("🏗️  Comprehensive Python Package Installation Script")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Current executable: {sys.executable}")
    
    # Handle virtual environment creation/management
    venv_manager = None
    python_executable = sys.executable
    
    if args.create_venv or args.venv_path != './datascience_env':
        venv_manager = VirtualEnvManager(args.venv_path)
        
        if args.create_venv:
            if not venv_manager.create_venv(force=args.force):
                print("❌ Failed to create virtual environment. Exiting.")
                sys.exit(1)
        
        # Use virtual environment Python
        python_executable = venv_manager.get_venv_python()
        print(f"🐍 Using Python: {python_executable}")
        
        if not Path(python_executable).exists():
            print(f"❌ Python executable not found: {python_executable}")
            print("Make sure the virtual environment is created properly.")
            sys.exit(1)
        
        print(f"🔧 To activate this environment manually, run:")
        print(f"   {venv_manager.get_activation_command()}")
    
    print("=" * 60)
    
    # Filter packages by category if specified
    packages_to_install = PACKAGES
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
        packages_to_install = filter_packages_by_category(PACKAGES, categories)
        print(f"📦 Installing {len(packages_to_install)} packages from categories: {', '.join(categories)}")
    
    # Create installer instance with appropriate Python executable
    installer = PackageInstaller(timeout=args.timeout, python_executable=python_executable)
    
    # Run installation
    results = installer.install_packages_batch(packages_to_install)
    
    # Final summary
    print("\n🎉 Installation process completed!")
    print(f"Check 'package_installation.log' for detailed logs.")
    
    if venv_manager and args.create_venv:
        print(f"\n🐍 Virtual environment created at: {venv_manager.env_path}")
        print(f"To activate: {venv_manager.get_activation_command()}")
        
        # Create activation script
        if os.name != 'nt':  # Unix-like systems
            activate_script = Path("activate_env.sh")
            activate_script.write_text(f"#!/bin/bash\n{venv_manager.get_activation_command()}\n")
            activate_script.chmod(0o755)
            print(f"📝 Created activation script: {activate_script}")
    
    if results['failed']:
        print("\n⚠️  Some packages failed to install. You may need to:")
        print("   • Check your internet connection")
        print("   • Install system dependencies")
        print("   • Consider using conda for failed packages")
        print("   • Check specific error messages in the log")
        
        # Exit with error code if critical packages failed
        critical_packages = ['numpy', 'pandas', 'matplotlib', 'scikit-learn']
        failed_critical = [pkg for pkg, _ in results['failed'] if pkg in critical_packages]
        
        if failed_critical:
            print(f"❌ Critical packages failed: {failed_critical}")
            sys.exit(1)
    
    print("✅ Installation script completed successfully!")


if __name__ == "__main__":
    main()