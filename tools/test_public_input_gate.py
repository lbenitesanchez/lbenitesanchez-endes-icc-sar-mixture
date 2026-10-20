"""Exercise the public-input allowlist and path guards without model fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_code_release


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    checks = []
    with tempfile.TemporaryDirectory(prefix='iaos_public_input_guard_') as temporary:
        fixture = (Path(temporary) / 'checkout').resolve()
        for source in check_code_release.files():
            destination = fixture / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        def seal():
            path = fixture / 'provenance/release_files.json'
            manifest = json.loads(path.read_text())
            excluded = set(manifest['excluded_from_self_index'])
            manifest['files_sha256'] = {
                p.relative_to(fixture).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(fixture.rglob('*'))
                if p.is_file() and not p.is_symlink() and p.relative_to(fixture).as_posix() not in excluded
            }
            path.write_text(json.dumps(manifest, indent=2) + '\n')

        def gate(label, finding=None):
            # Resealing the fixture ensures failures come from the CSV/path policy,
            # even if a newly added file has been added to the release inventory.
            seal()
            result = subprocess.run([sys.executable, 'tools/check_code_release.py'],
                                    cwd=fixture, env=env, capture_output=True, text=True)
            receipt = json.loads(result.stdout)
            if finding is None:
                assert result.returncode == 0 and receipt['approved_derived_csv_files'] == 37, receipt
                assert receipt['approved_simplified_csv_files'] == 35, receipt
                assert receipt['approved_model_auxiliary_csv_files'] == 2, receipt
            else:
                assert result.returncode != 0, label
                assert any(finding in row[1] for row in receipt['findings']), receipt
            checks.append(label)

        gate('complete approved bundle passes')
        extra = fixture / 'simplified/inputs/unregistered.csv'
        extra.write_text('value\n1\n')
        gate('unregistered CSV rejected even when inventoried', 'unapproved artifact extension')
        extra.unlink()

        registered = fixture / next(iter(check_code_release.approved_csvs()))
        original = registered.read_bytes()
        registered.write_bytes(original + b'\n')
        gate('changed registered CSV rejected even when inventoried', 'approved CSV hash differs')
        registered.unlink()
        gate('missing registered CSV rejected', 'missing approved CSV')
        registered.write_bytes(original)

        auxiliary = fixture / 'reproduction/inputs/aux_sikov14_official.csv'
        original_auxiliary = auxiliary.read_bytes()
        auxiliary.write_bytes(original_auxiliary + b'\n')
        gate('changed auxiliary CSV rejected even when inventoried', 'approved CSV hash differs')
        auxiliary.unlink()
        gate('missing auxiliary CSV rejected', 'missing approved CSV')
        auxiliary.write_bytes(original_auxiliary)

        review = fixture / 'provenance/principal_data_review.json'
        original_review = review.read_bytes()
        auxiliary_review = json.loads(original_review)
        auxiliary_review['approved_auxiliary_files']['reproduction/inputs/aux_sikov14_official.csv'] = '0' * 64
        review.write_text(json.dumps(auxiliary_review, indent=2) + '\n')
        gate('altered auxiliary allowlist hash rejected', 'reviewed two-file contract')
        auxiliary_review = json.loads(original_review)
        auxiliary_review['approved_auxiliary_files']['reproduction/inputs/unregistered.csv'] = '0' * 64
        review.write_text(json.dumps(auxiliary_review, indent=2) + '\n')
        gate('extra auxiliary allowlist path rejected', 'reviewed two-file contract')
        review.write_bytes(original_review)

        linked = fixture / 'linked_inputs'
        linked.symlink_to(fixture / 'simplified/inputs', target_is_directory=True)
        gate('directory symlink rejected', 'symlink')
        linked.unlink()
        linked = fixture / 'dangling_link'
        linked.symlink_to(fixture / 'absent')
        gate('dangling symlink rejected', 'symlink')
        linked.unlink()

        document = fixture / 'excluded_document.docx'
        document.write_bytes(b'\x00\xff')
        gate('binary document rejected even when inventoried', 'binary payload')
        document.unlink()
        gate('restored approved bundle passes')

        spec = importlib.util.spec_from_file_location('iaos_guard_runtime', fixture / 'simplified/support/runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        bundled = fixture / 'simplified/inputs'
        assert runtime._data_root(bundled) == bundled
        checks.append('exact bundled input root accepted')
        try:
            runtime._data_root(fixture / 'simplified')
        except ValueError:
            checks.append('other in-checkout input roots rejected')
        else:
            raise AssertionError('Other in-checkout input root was accepted')
        try:
            runtime._output_root(bundled, fixture / 'generated')
        except ValueError:
            checks.append('in-checkout output root rejected')
        else:
            raise AssertionError('In-checkout output root was accepted')
        external = (Path(temporary) / 'external_inputs').resolve()
        shutil.copytree(bundled, external)
        assert runtime.verify_inputs(external)['input_location'] == 'external'
        checks.append('external copy of approved bundle accepted')
        linked = Path(temporary) / 'input_link'
        linked.symlink_to(bundled, target_is_directory=True)
        try:
            runtime._data_root(linked)
        except ValueError:
            checks.append('linked input root rejected')
        else:
            raise AssertionError('Linked input root was accepted')

    print(json.dumps({'status': 'PASS_PUBLIC_INPUT_GUARDS', 'checks': checks,
                      'model_fits_run': False, 'original_inputs_modified': False}, indent=2))


if __name__ == '__main__':
    main()
