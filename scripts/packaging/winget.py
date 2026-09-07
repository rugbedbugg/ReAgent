"""Create Winget manifests for the exact installer validated by CI."""
import argparse
import hashlib
import re
from pathlib import Path


def generate(version, installer, output, url=None):
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('Expected a stable semantic version')
    checksum = hashlib.sha256(installer.read_bytes()).hexdigest().upper()
    url = url or f'https://github.com/rugbedbugg/ReAgent/releases/download/v{version}/{installer.name}'
    package = 'rugbedbugg.ReAgent'
    common = f'PackageIdentifier: {package}\nPackageVersion: {version}\n'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{package}.yaml').write_text(common + 'DefaultLocale: en-US\nManifestType: version\nManifestVersion: 1.6.0\n')
    (output / f'{package}.locale.en-US.yaml').write_text(common + '''PackageLocale: en-US
Publisher: Partha Pratim Gogoi
PackageName: ReAgent
License: Apache-2.0
ShortDescription: Evidence-grounded retrosynthesis planning and route scoring.
PackageUrl: https://github.com/rugbedbugg/ReAgent
LicenseUrl: https://github.com/rugbedbugg/ReAgent/blob/main/LICENSE
ManifestType: defaultLocale
ManifestVersion: 1.6.0
''')
    (output / f'{package}.installer.yaml').write_text(common + f'''InstallerType: inno
Scope: user
Installers:
- Architecture: x64
  InstallerUrl: {url}
  InstallerSha256: {checksum}
  ProductCode: ReAgent_is1
ManifestType: installer
ManifestVersion: 1.6.0
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--installer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url')
    args = parser.parse_args()
    generate(args.version, args.installer, args.output, args.url)
