"""Test-session defaults.

The journal: every `View.load()` writes a record to the active journal, which defaults to
`~/.geminga/journal.jsonl`. A test run must not leave fixture windows in the analyst's own
trail, so unless the environment already names one, the suite journals into a temporary file.
Tests about the journal itself open their own.
"""
import os
import tempfile

if "GEMINGA_JOURNAL" not in os.environ:
    os.environ["GEMINGA_JOURNAL"] = os.path.join(
        tempfile.mkdtemp(prefix="geminga-test-journal-"), "journal.jsonl")
