#!/usr/bin/env bash
set -euo pipefail
pacman -Syu --noconfirm --needed base-devel git namcap
useradd -m builder
# python311 is an AUR dependency, not Arch's system Python. Build the real
# dependency package before ReAgent, preserving the existing PKGBUILD contract.
printf 'builder ALL=(ALL) NOPASSWD: /usr/bin/pacman\n' > /etc/sudoers.d/builder
chmod 440 /etc/sudoers.d/builder
cache="$PWD/out/python311-cache"
mkdir -p "$cache"
if compgen -G "$cache/python311-*.pkg.tar.zst" > /dev/null; then
    pacman -U --noconfirm "$cache"/python311-*.pkg.tar.zst
else
    su builder -c 'git clone https://aur.archlinux.org/python311.git /home/builder/python311'
    su builder -c 'cd /home/builder/python311 && git checkout 204461e2219956562a4faeeded74d35f8e9fd5d5 && makepkg --syncdeps --install --noconfirm'
    cp /home/builder/python311/python311-*.pkg.tar.zst "$cache/"
fi
chown -R builder:builder prepared/aur
cd prepared/aur
su builder -c 'makepkg --printsrcinfo' > .SRCINFO
su builder -c 'makepkg --verifysource'
su builder -c 'makepkg --cleanbuild --noconfirm'
namcap PKGBUILD
namcap ./*.pkg.tar.zst
pacman -U --noconfirm ./*.pkg.tar.zst
reagent --help
reagent-download-data --help
pacman -R --noconfirm reagent
