#!/usr/bin/env bash
# Slurm batch script for a frozen run (freeze_source.sh): unpack the run's commit into
# node-local temporary space, run the command there, and remove the copy at the end.
#
#   sbatch [options] scripts/slurm/run_frozen.sh REPO COMMIT COMMAND [ARGS...]
#
# sbatch copies this script when it submits, so only the commit decides what runs. The
# command runs in the unpacked tree, which is also SLURM_SUBMIT_DIR, the directory every
# launcher puts on the import path; relative paths in ARGS resolve against it. ECAD_REPO
# and ECAD_SOURCE_COMMIT are exported: results record the commit, and a job that submits
# successors gives them the same commit. Signals Slurm sends this shell (USR1 before the
# time limit with --signal=B:USR1@..., TERM, INT) are passed on to the command.
set -euo pipefail
REPO="${1:?Usage: run_frozen.sh REPO COMMIT COMMAND [ARGS...]}"
COMMIT="${2:?Usage: run_frozen.sh REPO COMMIT COMMAND [ARGS...]}"
shift 2
(($#)) || { echo "run_frozen.sh: no command" >&2; exit 2; }
git -C "$REPO" cat-file -e "$COMMIT^{commit}"
SOURCE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ecad-source-${SLURM_JOB_ID:-local}-XXXXXX")"
trap 'rm -rf "$SOURCE_DIR"' EXIT
git -C "$REPO" archive "$COMMIT" | tar -x -C "$SOURCE_DIR"
cd "$SOURCE_DIR"
export SLURM_SUBMIT_DIR="$SOURCE_DIR" ECAD_REPO="$REPO" ECAD_SOURCE_COMMIT="$COMMIT"
echo "Frozen source $COMMIT unpacked in $SOURCE_DIR"
"$@" &
child=$!
for signal in USR1 TERM INT; do trap "kill -s $signal $child 2>/dev/null || true" "$signal"; done
status=0
while :; do
    wait "$child" && status=0 || status=$?
    # A trapped signal interrupts wait while the command is still running: wait again.
    kill -0 "$child" 2>/dev/null || break
done
# If the command ended just after a trapped signal, collect its own exit status.
if ((status > 128)); then
    wait "$child" 2>/dev/null && status=0 || { code=$?; ((code == 127)) || status=$code; }
fi
exit "$status"
