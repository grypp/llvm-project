import os, pathlib, subprocess, tempfile
root = pathlib.Path('/home/gozen/work/llvm-project')
cache = tempfile.mkdtemp(prefix='mlir-tutorial-cache-')
env = dict(os.environ, PYTHONPATH=str(root/'build/tools/mlir/python_packages/mlir_core'), PYTHONDONTWRITEBYTECODE='1', MLIR_DSL_CACHE_DIR=cache)
for key in ['MLIR_DSL_NO_CACHE', 'MLIR_DSL_DISABLE_FILE_CACHING']:
    env.pop(key, None)
for prefix in ['MISS','HIT']:
    p = subprocess.run(['/usr/bin/python3.12', 'mlir/test/python/dsl/file_cache.py'], cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['build/bin/FileCheck','mlir/test/python/dsl/file_cache.py','--check-prefix='+prefix],cwd=root,input=p.stdout,text=True,check=True)
    print(prefix, p.stdout)
print('CACHE', cache)
