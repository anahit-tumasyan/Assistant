import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Run_command as rc  # noqa: E402


class CheckCommandTests(unittest.TestCase):
    def assertAllowed(self, command):
        ok, reason = rc.check_command(command)
        self.assertTrue(ok, f"{command!r} should be allowed, got: {reason}")

    def assertBlocked(self, command):
        ok, _ = rc.check_command(command)
        self.assertFalse(ok, f"{command!r} should be blocked")

    def test_everyday_commands_are_allowed(self):
        for command in [
            "ls", "ls -la ~/Documents", "open ~/Downloads", "open -a 'Google Chrome'",
            "cd ~/Desktop", "cd ~/Desktop && ls", "ps aux | grep -i python",
            'find . -name "*.py" 2>/dev/null', "ls 2>&1 | head -n 5", "wc -l < data.csv",
            "git status", "du -sh ~/Downloads", "top -l 1 -n 10", 'grep -rn "a|b" .',
            'echo "Sorry, I\'m not allowed to delete anything."', "/bin/ls",
        ]:
            self.assertAllowed(command)

    def test_programs_off_the_list_are_blocked(self):
        for command in ["rm -rf ~", "sudo ls", "shutdown -h now", "kill 1234",
                        "FOO=1 ls", "/tmp/evil/ls", "./ls", ""]:
            self.assertBlocked(command)

    def test_chaining_cannot_smuggle_in_another_program(self):
        for command in ["ls; rm -rf ~", "ls && rm x", "ls || rm x", "ls | sh",
                        "ls a#; rm -rf ~", "echo hi #; rm -rf ~", "ls\nrm -rf ~",
                        "ls & rm x", "(rm x)", "ls |"]:
            self.assertBlocked(command)

    def test_substitution_and_writing_files_are_blocked(self):
        for command in ["echo $(whoami)", "echo `whoami`", "ls > out.txt",
                        "ls >> out.txt", "cat <<EOF", "echo 'unbalanced"]:
            self.assertBlocked(command)

    def test_dangerous_options_are_blocked(self):
        for command in ["find . -exec rm {} \\;", "find . -name x -delete",
                        "sed -i '' 's/a/b/' f.txt", "sed --in-place 's/a/b/' f.txt",
                        "sort -o out.txt in.txt", "rg --pre ./x foo"]:
            self.assertBlocked(command)

    def test_risky_commands_need_confirmation(self):
        for command in ["pip3 install requests", "brew install wget", "python3 x.py",
                        "curl -O https://example.com/a.zip", "git push",
                        "ps aux | awk '{print $1}'"]:
            self.assertTrue(rc.needs_confirmation(command), command)
        for command in ["ls", "git status", "git log --oneline -n 5",
                        "open ~/Downloads", "ps aux | grep python"]:
            self.assertFalse(rc.needs_confirmation(command), command)

    def test_endless_commands_get_a_count(self):
        segments, _ = rc._split_segments("ping google.com")
        self.assertEqual(rc._make_non_interactive("ping google.com", segments),
                         "ping -c 4 google.com")
        segments, _ = rc._split_segments("ping -c 2 google.com")
        self.assertEqual(rc._make_non_interactive("ping -c 2 google.com", segments),
                         "ping -c 2 google.com")
        if sys.platform == "darwin":
            segments, _ = rc._split_segments("top | head")
            self.assertEqual(rc._make_non_interactive("top | head", segments), "top -l 1 | head")


class RunCommandTests(unittest.TestCase):
    def setUp(self):
        self._saved_cwd = rc._cwd
        self.tmp = tempfile.TemporaryDirectory()
        rc._cwd = os.path.realpath(self.tmp.name)

    def tearDown(self):
        rc._cwd = self._saved_cwd
        self.tmp.cleanup()

    def test_returns_what_the_command_printed(self):
        self.assertEqual(rc.run_command("echo hello").strip(), "hello")

    def test_pipes_work(self):
        self.assertEqual(rc.run_command("printf 'b\\na\\n' | sort").split(), ["a", "b"])

    def test_blocked_command_never_runs(self):
        marker = os.path.join(rc._cwd, "marker")
        output = rc.run_command(f"touch {marker}; rm -rf {marker}")
        self.assertTrue(output.startswith("Blocked"), output)
        self.assertFalse(os.path.exists(marker))

    def test_errors_are_reported(self):
        self.assertIn("Error (exit code", rc.run_command("ls /definitely/not/here"))

    def test_cd_is_remembered(self):
        os.mkdir(os.path.join(rc._cwd, "inner"))
        self.assertIn("Now in", rc.run_command("cd inner"))
        self.assertEqual(rc.run_command("pwd").strip(), os.path.join(rc._cwd))
        self.assertTrue(rc.current_directory().endswith("inner"))

    def test_cd_to_a_missing_folder(self):
        self.assertIn("no such folder", rc.run_command("cd nope"))

    def test_cd_alone_goes_home(self):
        rc.run_command("cd")
        self.assertEqual(rc.current_directory(), os.path.expanduser("~"))

    def test_slow_pipeline_is_stopped_without_hanging(self):
        start = time.monotonic()
        output = rc.run_command("ping -c 50 -i 1 127.0.0.1 | grep -c icmp", timeout=1.5)
        self.assertIn("stopped after", output)
        self.assertLess(time.monotonic() - start, 5)


if __name__ == "__main__":
    unittest.main()
