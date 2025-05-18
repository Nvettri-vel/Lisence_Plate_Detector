git clone https://github.com/rathaROG/lapx.git
cd lapx
python -m pip install --upgrade pip
pip install "setuptools>=67.2.0"
pip install wheel build
python -m build --wheel
cd dist