#!/usr/bin/env bash
# Freeze the source a Slurm run uses, as a git commit under refs/runs/NAME/<UTC time>, and
# print the commit. Jobs then run that commit through run_frozen.sh, so later edits to the
# working tree cannot reach a queued or running job.
#
#   freeze_source.sh [options] NAME
#
# The commit holds the working tree's ecad-dlm/ and scripts/ as they are now, committed or
# not, untracked files included and ignored files left out. It is written through a
# temporary index: the branch, the real index and the files are untouched, and nothing
# shows in `git log`. Refs under refs/runs are local; `git push` does not send them.
#
#   --include PATH   also freeze PATH (ignored files left out)
#   --force PATH     also freeze PATH even though it is ignored (e.g. Dream/, the upstream
#                    checkout); __pycache__ and *.pyc are skipped
#   --file DEST=SRC  freeze file SRC at path DEST in the tree (e.g. config.yaml=some.yaml)
#   --base COMMIT    start from COMMIT's tree instead of the working tree's ecad-dlm/ and
#                    scripts/; only --include, --force and --file are added to it
#   --again          freeze again although NAME was frozen before (runs that resume)
#   --reuse          print NAME's latest frozen commit if there is one; freeze otherwise
#   --existing       print NAME's latest frozen commit; fail if there is none
#
# Without --again, --reuse or --existing, a NAME frozen before is refused, which guards
# against submitting the same run twice. List runs with
#   git for-each-ref refs/runs
# and see a run's code with `git show COMMIT:PATH` or `git archive COMMIT | tar -x -C DIR`.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mode=new base="" include=(ecad-dlm scripts) force=() files=()
while (($#)); do
    case "$1" in
        --include) include+=("${2:?}"); shift 2 ;;
        --force) force+=("${2:?}"); shift 2 ;;
        --file) files+=("${2:?}"); shift 2 ;;
        --base) base="${2:?}"; include=(); shift 2 ;;
        --again|--reuse|--existing) mode="${1#--}"; shift ;;
        --) shift; break ;;
        -*) echo "freeze_source.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
NAME="${1:?Usage: freeze_source.sh [options] NAME}"
git check-ref-format "refs/runs/$NAME/x" || { echo "Not a valid run name: $NAME" >&2; exit 2; }
latest="$(git -C "$REPO" for-each-ref --sort=-refname --count=1 --format='%(objectname)' "refs/runs/$NAME/")"
case "$mode" in
    reuse) [[ -n "$latest" ]] && { echo "$latest"; exit 0; } ;;
    existing) [[ -n "$latest" ]] || { echo "Run $NAME was never frozen" >&2; exit 1; }; echo "$latest"; exit 0 ;;
    new) [[ -z "$latest" ]] || { echo "Run $NAME was already frozen ($latest); refusing duplicate submission" >&2; exit 1; } ;;
esac
# Resolve file arguments against the caller's directory before git -C changes it.
for index in "${!files[@]}"; do
    source_path="${files[$index]#*=}"
    [[ "${files[$index]}" == *=* && -f "$source_path" ]] || { echo "--file needs DEST=SRC with an existing SRC: ${files[$index]}" >&2; exit 2; }
    files[$index]="${files[$index]%%=*}=$(realpath "$source_path")"
done
GIT_INDEX_FILE="$(mktemp -u "${TMPDIR:-/tmp}/freeze-index-XXXXXX")"
export GIT_INDEX_FILE
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
cd "$REPO"
if [[ -n "$base" ]]; then
    git read-tree "$base^{tree}"
fi
((${#include[@]})) && git add -A -- "${include[@]}"
for path in "${force[@]}"; do
    [[ -e "$path" ]] || { echo "--force path does not exist: $path" >&2; exit 2; }
    find "$path" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z |
        while IFS= read -r -d '' file; do
            printf '100644 %s\t%s\n' "$(git hash-object -w -- "$file")" "$file"
        done | git update-index --add --index-info
done
for pair in "${files[@]}"; do
    git update-index --add --cacheinfo "100644,$(git hash-object -w -- "${pair#*=}"),${pair%%=*}"
done
tree="$(git write-tree)"
parent="$(git rev-parse -q --verify HEAD || true)"
commit="$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "Frozen source for run $NAME")"
git update-ref "refs/runs/$NAME/$(date -u +%Y%m%dT%H%M%S.%NZ)" "$commit" ""
echo "$commit"
