from setuptools import find_packages, setup 

# Read requirements.txt for dependencies
with open("requirements.txt", "r") as f:
  requirements = f.readlines()
print(requirements)

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setup(
  name = "mlops_modular_project",
  version="0.1.0",
  author="Antar Chandra Nath",
  author_email="antarnath.cse@gmail.com",
  description = "A modular MLOps project structure",
  long_description = long_description,
  long_description_content_type = "text/markdown",
  url="https://github.com/antarnath/Modular-Workflow-and-Project-Setup-Basics",
  packages=find_packages(),
  classifiers=[
    "Development Status :: 3 - Beta",
    "Intended Audience :: ML Engineers",
    "Programming Language :: Python >= 3.8", 
  ],
  python_requires=">=3.8",
  install_requires=requirements,
  extras_require={
    "dev": [
      'pytest>=7.1.1',
      'pytest-cov>=2.12.1',
      'flake8>=3.9.0',
      'black>=22.3.0',
      'isort>=5.10.1',
      'mypy>=0.942'
    ]
  }
)
