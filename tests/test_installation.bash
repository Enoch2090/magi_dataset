rm ./bin ./build ./dist ./include ./lib ./share ./*.egg-info ./*/__pycache__ -r
rm ./pyvenv.cfg
source ~/.bashrc
python3 -m venv .
source ./bin/activate
pip install wheel
python3 setup.py bdist_wheel
pip install ./dist/magi_dataset-1.0.0-py3-none-any.whl