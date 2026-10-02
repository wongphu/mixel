#!/bin/bash
# Prepares a mixel release, up to a local commit and tag:
#
#   scripts/release.sh 0.4.2               # build, test, package, commit, tag
#   scripts/release.sh 0.4.2 --no-models   # skip what needs the model weights
#
# It bumps the version, builds for macOS 14 and up, runs the tests
# (with --include-ignored, so with the real models, unless --no-models),
# packages mixel and mlx.metallib into target/dist/, checks the tarball from a
# fresh folder, then commits "Release <version>" and tags v<version>. It
# publishes nothing: the commands to push and create the GitHub release are
# printed at the end.
set -euo pipefail

# MLX's minimum. Building for the build machine's macOS (MLX's default) makes
# the binary and its GPU kernels need that exact version or later. On 14.0,
# MLX leaves out its NAX kernels (for M5 GPUs; they need 26.2): M5 owners can
# build from source.
MIN_MACOS=14.0

VERSION="${1:-}"
MODELS=1
[ "${2:-}" = "--no-models" ] && MODELS=0
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
  exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
NAME="mixel-$VERSION-macos-arm64"
DIST="$REPO/target/dist"
step() { printf '\n== %s\n' "$*"; }
fail() { echo "error: $*" >&2; exit 1; }

step "Checks"
[ "$(git branch --show-current)" = main ] || fail "not on main"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "uncommitted changes"
git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null && fail "tag v$VERSION exists"
git ls-remote --exit-code --tags origin "v$VERSION" >/dev/null 2>&1 && fail "tag v$VERSION exists on origin"
echo "Releasing $VERSION (now $(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)) for macOS $MIN_MACOS and up"

step "Version"
trap 'echo "failed: undo the version bump with git checkout Cargo.toml Cargo.lock" >&2' ERR
perl -0pi -e "s/^version = \"[^\"]+\"/version = \"$VERSION\"/m" Cargo.toml

# A separate target directory, and its own place for MLX's metallib: by
# default the build installs it into ~/.mlx/lib/<hash>/, shared with (and
# overwriting) the development build's.
export MACOSX_DEPLOYMENT_TARGET="$MIN_MACOS"
export CARGO_TARGET_DIR="$REPO/target/release-dist"
export MLX_RS_METAL_PATH="$CARGO_TARGET_DIR/metal"

step "Build"
cargo build --release
BIN="$CARGO_TARGET_DIR/release/mixel"
METALLIB="$MLX_RS_METAL_PATH/mlx.metallib"
[ -f "$METALLIB" ] || fail "no $METALLIB"
minos="$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/ {f = 1} f && $1 == "minos" {print $2; exit}')"
[ "$minos" = "$MIN_MACOS" ] || fail "mixel is built for macOS $minos, not $MIN_MACOS"
mlx_target="$(cat "$CARGO_TARGET_DIR"/release/build/mlx-sys-*/output | sed -n 's/^-- Building for macOS //p' | tail -1)"
[ "$mlx_target" = "$MIN_MACOS" ] || fail "MLX is built for macOS $mlx_target, not $MIN_MACOS"
[ "$("$BIN" --version)" = "mixel $VERSION" ] || fail "mixel --version: $("$BIN" --version)"
echo "mixel $VERSION, binary and MLX for macOS $MIN_MACOS"

step "Tests"
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
if [ $MODELS -eq 1 ]; then
  caffeinate -s cargo test --release -- --include-ignored
else
  cargo test --release
fi

step "Package"
rm -rf "${DIST:?}/$NAME" "$DIST/$NAME.tar.gz" "$DIST/$NAME.tar.gz.sha256"
mkdir -p "$DIST/$NAME"
cp "$BIN" "$METALLIB" "$DIST/$NAME/"
cat >"$DIST/$NAME/README.txt" <<EOF
mixel $VERSION for Apple Silicon Macs (arm64), macOS $MIN_MACOS or later

Keep mlx.metallib in the same folder as mixel: it holds MLX's GPU kernels.

  ./mixel --help
  ./mixel --prompt "a red fox in fresh snow" --seed 1 --output fox.png
  ./mixel --quantize 4 --prompt "a red fox in fresh snow" --seed 1 --output fox.png

The first run downloads the model weights (~33 GB) to ~/.cache/huggingface.
--quantize 8 or 4 needs less memory (5.4 GB for a 1024x1024 image at 4 bits);
on a 16 GB Mac, use --quantize 8 for z-image-turbo and 4 for the qwen models.
If macOS blocks the downloaded binary, run:
  xattr -d com.apple.quarantine mixel
EOF
(cd "$DIST" && tar czf "$NAME.tar.gz" "$NAME" && shasum -a 256 "$NAME.tar.gz" >"$NAME.tar.gz.sha256")

step "Check the tarball from a fresh folder"
CHECK="$(mktemp -d -t mixel-release)"
# Hide the build's metallib, so mixel can only use the one in the tarball.
mv "$MLX_RS_METAL_PATH" "$MLX_RS_METAL_PATH.hidden"
restore() { [ -d "$MLX_RS_METAL_PATH.hidden" ] && mv "$MLX_RS_METAL_PATH.hidden" "$MLX_RS_METAL_PATH"; rm -rf "$CHECK"; }
trap restore EXIT
(cd "$DIST" && shasum -a 256 -c "$NAME.tar.gz.sha256")
tar xzf "$DIST/$NAME.tar.gz" -C "$CHECK"
[ "$("$CHECK/$NAME/mixel" --version)" = "mixel $VERSION" ] || fail "packaged mixel --version"
if [ $MODELS -eq 1 ]; then
  "$CHECK/$NAME/mixel" --quantize 4 --prompt "a red fox in fresh snow" --seed 1 \
    --width 256 --height 256 --num-steps 2 --output "$CHECK/fox.png" | tail -1
  [ -s "$CHECK/fox.png" ] || fail "the packaged mixel made no image"
fi
restore
trap - EXIT

step "Commit and tag"
git commit -q -m "Release $VERSION" Cargo.toml Cargo.lock
git tag -a "v$VERSION" -m "mixel $VERSION"
git log --oneline -1

cat <<EOF

Ready: $DIST/$NAME.tar.gz (and .sha256), commit and tag v$VERSION (local).
To publish, write the release notes, then:

  git push origin main && git push origin v$VERSION
  gh release create v$VERSION --verify-tag --title "mixel $VERSION" --notes-file NOTES.md \\
    "$DIST/$NAME.tar.gz" "$DIST/$NAME.tar.gz.sha256"
EOF
