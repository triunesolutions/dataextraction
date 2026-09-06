"""One-shot Kaggle runner: clone, credentials, cleanup, run, report.

Driven from a Kaggle notebook with a single line, so the notebook never has to
be edited again when this logic changes:

    import urllib.request as u; exec(u.urlopen(
        'https://raw.githubusercontent.com/triunesolutions/dataextraction/'
        'merge/main-pranali-sahil/notebooks/kaggle_run.py').read().decode())

Safe to re-run. Every step that bit us in a real session is handled here rather
than left to cell ordering:

  * chdir out of the repo before deleting it -- deleting the directory the
    process is standing in leaves every subprocess with a dangling cwd, which
    surfaces as a confusing "unable to read current working directory".
  * credentials go to the environment, not a .env inside the repo, because the
    re-clone deletes that file and the run then fails far from the cause.
  * rows a broken run cached as 'ok' with zero equipment are cleared, so the
    content-hash skip cannot make a failure permanent.

Override before exec() if needed: KAGGLE_RUN_LIMIT, KAGGLE_RUN_BRANCH.
"""
import glob
import os
import shutil
import sqlite3
import subprocess
import sys

REPO_URL = "https://github.com/triunesolutions/dataextraction"
BRANCH = os.environ.get("KAGGLE_RUN_BRANCH", "merge/main-pranali-sahil")
LIMIT = int(os.environ.get("KAGGLE_RUN_LIMIT", "0"))  # 0 = every file


def _platform():
    """'kaggle' or 'colab'. Both are supported because the two differ only in
    where files live and how secrets are read -- nothing about the extraction."""
    if os.path.isdir("/kaggle/input") or os.path.isdir("/kaggle/working"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "kaggle"


PLATFORM = _platform()

if PLATFORM == "colab":
    # Prefer Drive when it is mounted. This is the substantive reason to use
    # Colab: a Kaggle session discards /kaggle/working unless Save Version is
    # pressed, and two runs' results were lost that way. Anything written under
    # Drive survives the session ending, and the database is the valuable part
    # -- it is what lets a later run skip finished files by content hash.
    _drive = "/content/drive/MyDrive"
    WORK = os.path.join(_drive, "hvac_extract") if os.path.isdir(_drive) else "/content"
    os.makedirs(WORK, exist_ok=True)
    # Corpus: set KAGGLE_RUN_CORPUS, else look in Drive, else /content.
    CORPUS_ROOTS = [os.environ.get("KAGGLE_RUN_CORPUS", ""),
                    os.path.join(_drive, "hvac_pdfs"), "/content"]
else:
    WORK = "/kaggle/working"
    CORPUS_ROOTS = sorted(glob.glob("/kaggle/input/*"))

REPO_DIR = os.path.join(WORK, "dataextraction")
DB = os.path.join(WORK, "hvac.db")
CSV = os.path.join(WORK, "hvac.csv")


def _get_secret(name):
    """Read an API key from whichever secret store this platform provides."""
    if PLATFORM == "colab":
        from google.colab import userdata
        return userdata.get(name)
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret(name)


def _step(msg):
    print("\n=== %s ===" % msg, flush=True)


# --------------------------------------------------------------------------- #
_step("platform")
print(PLATFORM, "| work dir:", WORK)

_step("code")
os.chdir(WORK)                       # never stand inside what we delete
shutil.rmtree(REPO_DIR, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR],
               cwd=WORK, check=True, capture_output=True)
print(subprocess.run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR,
                     capture_output=True, text=True).stdout.strip())

_step("dependencies")
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-r", "requirements.txt"],
               cwd=REPO_DIR, check=True)
import importlib
for m in ("pypdf", "pdfplumber", "pydantic", "dotenv"):
    importlib.invalidate_caches()
    assert importlib.util.find_spec(m), "missing dependency: %s" % m
import pdfplumber
print("pdfplumber", pdfplumber.__version__)

_step("credentials")
key = (_get_secret("GROQ_API_KEY") or "").strip()
assert key.startswith("gsk_"), "GROQ_API_KEY secret missing or malformed"
os.environ["MODEL_BACKEND"] = "groq"
os.environ["GROQ_API_KEY"] = key
os.environ["GROQ_MODEL"] = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
stale = os.path.join(REPO_DIR, ".env")
if os.path.exists(stale):
    os.remove(stale)
print("set in environment, length %d (not printed)" % len(key))

_step("database")
prior = (glob.glob("/kaggle/input/*/hvac.db") + glob.glob("/kaggle/input/*/*/hvac.db")
         if PLATFORM == "kaggle" else [])
if prior and not os.path.exists(DB):
    shutil.copy(prior[0], DB)
    print("restored from", prior[0])
if os.path.exists(DB):
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    bad = ("SELECT file_hash FROM files WHERE status='ok' AND file_hash NOT IN "
           "(SELECT file_hash FROM equipment)")
    try:
        n = len(con.execute(bad).fetchall())
        for t in ("equipment", "project_team", "projects", "files"):
            con.execute("DELETE FROM %s WHERE file_hash IN (%s)" % (t, bad))
        con.commit()
        print("cleared %d file(s) cached as 'ok' with zero rows" % n)
    except sqlite3.Error as e:
        print("skipped cleanup:", e)
    con.close()
else:
    print("fresh database")

_step("corpus")
CORPUS = None
for root in CORPUS_ROOTS:
    if root and os.path.isdir(root) and glob.glob(
            os.path.join(root, "**", "*.pdf"), recursive=True):
        CORPUS = root
        break
assert CORPUS, ("no PDFs found in %s -- attach the dataset (Kaggle) or set "
                "KAGGLE_RUN_CORPUS to the folder holding them (Colab)" % CORPUS_ROOTS)
print(CORPUS, "-", len(glob.glob(os.path.join(CORPUS, "**", "*.pdf"), recursive=True)), "PDFs")

_step("run")
cmd = [sys.executable, "run.py", CORPUS, "--db", DB, "--export", CSV]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
print(" ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print("exit code:", proc.returncode)

_step("report")
if os.path.exists(DB):
    con = sqlite3.connect(DB)
    print("status:")
    for r in con.execute("SELECT status, COUNT(*) FROM files GROUP BY status"):
        print("   %-20s %d" % r)

    print("\nper file:")
    q = ("SELECT f.filename, f.status, (SELECT COUNT(*) FROM equipment e "
         "WHERE e.file_hash=f.file_hash), p.project_name, p.location "
         "FROM files f LEFT JOIN projects p ON p.file_hash=f.file_hash "
         "ORDER BY f.filename")
    for name, status, rows, proj, loc in con.execute(q):
        print("   %-18s %-18s %4d rows  %-34s %s"
              % (name[:18], status, rows, (proj or "-")[:34], loc or "-"))

    total = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0] or 1
    print("\nfield coverage (of %d projects):" % total)
    for c in ("project_name", "location", "engineer"):
        got = con.execute("SELECT COUNT(%s) FROM projects" % c).fetchone()[0]
        print("   %-14s %d/%d  (%.0f%%)" % (c, got, total, got / total * 100))

    empty = con.execute("SELECT COUNT(*) FROM files WHERE status='ok' AND file_hash "
                        "NOT IN (SELECT file_hash FROM equipment)").fetchone()[0]
    print("\nfiles reporting 'ok' with zero rows: %d%s"
          % (empty, "  <-- investigate" if empty else ""))
    con.close()

for f in sorted(glob.glob(os.path.join(WORK, "*.csv")) + [DB]):
    if os.path.exists(f):
        print("   %8.2f MB  %s" % (os.path.getsize(f) / 1e6, f))
if PLATFORM == "kaggle":
    print("\nSave Version to keep these, then attach the output next session.")
else:
    print("\nWritten under %s -- survives the session ending." % WORK)
