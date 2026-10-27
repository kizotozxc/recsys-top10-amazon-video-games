"""Execute the complete notebook using this Python environment; save outputs in place."""
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parent
    kernels = root / '.cache' / 'jupyter' / 'kernels'
    spec = kernels / 'recsys-homework'
    spec.mkdir(parents=True, exist_ok=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python (recsys-homework)', 'language': 'python',
    }))
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['MPLCONFIGDIR'] = str(root / '.cache' / 'matplotlib')
    os.environ['IPYTHONDIR'] = str(root / '.cache' / 'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.cache' / 'jupyter' / 'runtime')
    path = root / 'experiments.ipynb'
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []; cell.execution_count = None
    def save_cell(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        print(f'Completed cell {cell_index+1}/{len(notebook.cells)}', flush=True)
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                print(out.get('text', '')[-2500:], flush=True)
    manager = KernelManager(kernel_name='recsys-homework',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = NotebookClient(notebook, km=manager, timeout=7200,
                            resources={'metadata': {'path': str(root)}},
                            on_cell_executed=save_cell)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f'Executed notebook saved: {path}', flush=True)


if __name__ == '__main__':
    main()
