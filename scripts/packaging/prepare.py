"""Generate release-based ReAgent channel inputs without changing tracked recipes."""
import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = 'rugbedbugg/ReAgent'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ReAgent-packaging'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def prepare(tag, output):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
        raise ValueError('Use an existing stable vMAJOR.MINOR.PATCH release tag')
    release = json.loads(fetch(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}'))
    if release['draft'] or release['prerelease'] or release['tag_name'] != tag:
        raise ValueError('Only published stable releases are supported')
    version = tag[1:]
    assets = {asset['name']: asset for asset in release['assets']}
    hashes = {}
    files = {}
    for name in [f'reagent-{version}-py3-none-any.whl', f'reagent-{version}.tar.gz']:
        asset = assets[name]
        content = fetch(asset['browser_download_url'])
        checksum = hashlib.sha256(content).hexdigest()
        if asset.get('digest') != 'sha256:' + checksum:
            raise ValueError(f'{name} does not match its GitHub release digest')
        hashes[name] = checksum
        files[name] = content
    installer_name = f'reagent-{version}-windows-x86_64-setup.exe'
    existing_installer = None
    if installer_name in assets:
        asset = assets[installer_name]
        existing_installer = fetch(asset['browser_download_url'])
        digest = hashlib.sha256(existing_installer).hexdigest()
        if asset.get('digest') != 'sha256:' + digest:
            raise ValueError('Existing Windows installer digest mismatch')
    output.mkdir(parents=True, exist_ok=True)
    choco = output / 'chocolatey'
    shutil.copytree('SUBMISSIONS/chocolatey', choco, dirs_exist_ok=True)
    ns = {'n': 'http://schemas.microsoft.com/packaging/2011/08/nuspec.xsd'}
    # Preserve the nuspec's existing namespace across supported schema versions.
    spec = choco / 'reagent.nuspec'
    tree = ET.parse(spec)
    namespace = tree.getroot().tag.split('}')[0].lstrip('{')
    ns['n'] = namespace
    ET.register_namespace('', namespace)
    tree.find('n:metadata/n:version', ns).text = version
    tree.find('n:metadata/n:releaseNotes', ns).text = release['html_url']
    tree.write(spec, encoding='utf-8', xml_declaration=True)
    wheel = f'reagent-{version}-py3-none-any.whl'
    install = choco / 'tools/chocolateyinstall.ps1'
    text = install.read_text()
    text = re.sub(r'reagent-[0-9.]+-py3-none-any.whl', wheel, text)
    text = re.sub(r'/download/v[0-9.]+/', f'/download/{tag}/', text)
    text = re.sub(r"checksum\s*= '[^']+'", f"checksum = '{hashes[wheel]}'", text)
    install.write_text(text)
    aur = output / 'aur'
    shutil.copytree('SUBMISSIONS/aur', aur, dirs_exist_ok=True)
    recipe = (aur / 'PKGBUILD').read_text()
    recipe = re.sub(r'^pkgver=.*$', f'pkgver={version}', recipe, flags=re.M)
    recipe = re.sub(r'^pkgrel=.*$', 'pkgrel=1', recipe, flags=re.M)
    recipe = re.sub(r"sha256sums=\('[^']+'", f"sha256sums=('{hashes[f'reagent-{version}.tar.gz']}'", recipe, count=1)
    (aur / 'PKGBUILD').write_text(recipe)
    payload = output / 'winget/payload'
    payload.mkdir(parents=True, exist_ok=True)
    (payload / wheel).write_bytes(files[wheel])
    if existing_installer is not None:
        (output / 'winget' / installer_name).write_bytes(existing_installer)
    shutil.copyfile(choco / 'tools/requirements.txt', payload / 'requirements.txt')
    (output / 'release.json').write_text(json.dumps({'tag': tag, 'version': version, 'repository': REPO, 'hashes': hashes}, indent=2) + '\n')
    print(f'Prepared {tag} under {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, default=Path('out/packaging'))
    args = parser.parse_args()
    prepare(args.tag, args.output)
