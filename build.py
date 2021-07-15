import os
import sys

python_executable = sys.executable
os.system('git clone https://github.com/xinntao/BasicSR.git')
os.chdir('BasicSR')
os.system(f'{python_executable} -m pip install -r requirements.txt')
os.system(f'BASICSR_EXT=True {python_executable} setup.py develop')
