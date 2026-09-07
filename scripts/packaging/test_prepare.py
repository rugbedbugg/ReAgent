import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare import prepare
from winget import generate


class PreparationTests(unittest.TestCase):
    def test_invalid_tag_does_not_download(self):
        with patch('prepare.fetch') as fetch:
            with self.assertRaises(ValueError):
                prepare('v1.2.3;bad', Path('unused'))
            fetch.assert_not_called()

    def test_release_digest_mismatch_is_rejected(self):
        release = {'draft': False, 'prerelease': False, 'tag_name': 'v1.2.3', 'assets': [
            {'name': 'reagent-1.2.3-py3-none-any.whl', 'browser_download_url': 'wheel', 'digest': 'sha256:bad'}]}
        with patch('prepare.fetch', side_effect=[json.dumps(release).encode(), b'bad wheel']):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    prepare('v1.2.3', Path(temp))
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_winget_uses_exact_installer_bytes_and_local_url_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installer = root / 'reagent-1.2.3-windows-x86_64-setup.exe'
            installer.write_bytes(b'installer bytes')
            output = root / 'manifests'
            generate('1.2.3', installer, output)
            data = (output / 'rugbedbugg.ReAgent.installer.yaml').read_text()
            self.assertIn(hashlib.sha256(b'installer bytes').hexdigest().upper(), data)
            self.assertIn('/download/v1.2.3/reagent-1.2.3-windows-x86_64-setup.exe', data)
            self.assertIn('InstallerType: inno', data)
            generate('1.2.3', installer, output, 'http://127.0.0.1:8765/setup.exe')
            self.assertIn('http://127.0.0.1:8765/setup.exe', (output / 'rugbedbugg.ReAgent.installer.yaml').read_text())


if __name__ == '__main__':
    unittest.main()
