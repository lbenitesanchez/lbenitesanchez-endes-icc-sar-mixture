"""Install the observed competitor environment locally, on explicit execution."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]


def main():
    if sys.version_info < (3,12):
        raise RuntimeError('Use Python3.12 or a newer compatible interpreter; the observed environment was3.12.')
    p = argparse.ArgumentParser()
    p.add_argument('--cmdstan', action='store_true', help='Download and compile CmdStan2.40.0 locally.')
    p.add_argument('--cores', type=int, default=1)
    a = p.parse_args()
    environment = ROOT / '.competitor_env'
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r',
                    str(Path(__file__).with_name('requirements_competitors.txt'))], check=True)
    if a.cmdstan:
        env = os.environ.copy()
        sdk = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
        if sys.platform == 'darwin' and sdk.exists():
            env.setdefault('SDKROOT', str(sdk))
        code = ('import cmdstanpy; from pathlib import Path; '
                f'p=Path({str(environment / "cmdstan")!r}); p.mkdir(exist_ok=True); '
                f'ok=cmdstanpy.install_cmdstan(version="2.40.0",dir=str(p),cores={a.cores},overwrite=False); '
                'assert ok,"CmdStan installation/build failed; see compiler output"')
        subprocess.run([str(python), '-c', code], env=env, check=True)
    print('Local environment created. No global Python or SDK settings changed.')


if __name__ == '__main__':
    main()
