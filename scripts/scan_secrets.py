"""Pre-commit secret scan for staged files. Exit code 1 (abort) if anything suspicious is staged.

    git add -A && python scripts/scan_secrets.py && git commit -m "..."

Flags: the Groq key prefix, Postgres URLs with an embedded user:password, Neon password
prefixes, and any staged database file (*.db, *.sqlite, *.db.bak). The prefixes are built from
parts below so this file does not flag itself.
"""

import re
import subprocess
import sys

GROQ_PREFIX = "gs" + "k_"
NEON_PREFIX = "np" + "g_"
PATTERNS = {
    "Groq key prefix": re.compile(re.escape(GROQ_PREFIX)),
    "Postgres URL with credentials": re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^\s/:@]+:[^\s@]+@"),
    "Neon password": re.compile(re.escape(NEON_PREFIX) + r"[A-Za-z0-9]{6,}"),
}
DB_FILE = re.compile(r"\.(db|sqlite|db\.bak)$", re.IGNORECASE)


def main() -> int:
    files = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                           capture_output=True, text=True, check=True).stdout.split()
    problems: list[str] = []
    for path in files:
        if DB_FILE.search(path) or path.endswith(".env"):
            problems.append(f"{path}: database/env file must not be committed")
            continue
        blob = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True).stdout
        text = blob.decode("utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    problems.append(f"{path}:{lineno}: {name}")
    print(f"scanned {len(files)} staged files")
    if problems:
        print("ABORT: possible secrets staged:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: no secrets or database files staged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
