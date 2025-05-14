"""
LLM Prompt Generator for Sublime Text
====================================
🧠 Smarter relevance scoring + sticky picker cursor
--------------------------------------------------
The top-5 files are still pre-selected, but their order and the relevance score
are now calculated with a richer heuristic:

```
score = (100 × distinct_identifier_hits)
        + (2 × total_identifier_hits)
        + (20 if file is in the same directory as the snippet)
        – (file_size_kB)
```

This favours files that:
* mention **many different** tokens from the selection,
* mention them **often**,
* live **nearby** in the project tree,
* aren’t gigantic.

The Quick Panel keeps your caret on the toggled item thanks to the
`selected_index` arg when it reopens.

(Fully vibe-coded)
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Dict, List, Set, Tuple

import sublime
import sublime_plugin

_SETTINGS_KEY = "llm_prompt_view_id"  # remember prompt view for multi-shot
_MONO = sublime.MONOSPACE_FONT

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _extract_identifiers(code: str) -> Set[str]:
    """Return relevant identifiers in *code* (very lightweight)."""
    blacklist = {
        "if",
        "for",
        "while",
        "switch",
        "case",
        "else",
        "return",
        "break",
        "continue",
        "try",
        "except",
        "catch",
        "finally",
        "def",
        "class",
        "struct",
        "enum",
        "namespace",
        "public",
        "private",
        "protected",
        "import",
        "from",
        "package",
        "do",
        "new",
        "delete",
        "this",
    }
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code))
    return {t for t in tokens if t not in blacklist and not t.isdigit()}


def _syntax_from_extension(path: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".sh": "bash",
    }.get(os.path.splitext(path)[1].lower(), "")


# ---------------------------------------------------------------------------
#  Very-heuristic definition extractor (for overview)
# ---------------------------------------------------------------------------


def _extract_definitions(text: str, language: str) -> Dict[str, List[Tuple[str, str]]]:
    classes: Dict[str, List[Tuple[str, str]]] = {}

    if language in {"python", "ruby"}:
        class_pat = re.compile(r"^\s*class\s+(\w+)")
        func_pat = re.compile(
            r"^\s*def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?", re.M
        )
    elif language in {"javascript", "typescript", "php", "java", "csharp"}:
        class_pat = re.compile(r"\bclass\s+(\w+)")
        func_pat = re.compile(r"(?:^|\s)(?:function\s+)?(\w+)\s*\(([^)]*)\)")
    else:
        class_pat = re.compile(r"\bclass\s+(\w+)")
        func_pat = re.compile(r"\b(\w+)\s*\(([^)]*)\)")

    for m in class_pat.finditer(text):
        classes.setdefault(m.group(1), [])

    for m in func_pat.finditer(text):
        name, args = m.group(1), m.group(2)
        if name in classes:
            continue
        classes.setdefault("", []).append((name, f"({args})"))

    return classes


# ---------------------------------------------------------------------------
#  Main command – generate prompt with smarter scoring
# ---------------------------------------------------------------------------


class GenerateLlmPromptCommand(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit) -> None:  # noqa: D401
        window = self.view.window()
        if window is None:
            return

        # 1. Capture selection -------------------------------------------------
        selections = [self.view.substr(r) for r in self.view.sel() if not r.empty()]
        if not selections:
            selections = [self.view.substr(sublime.Region(0, self.view.size()))]
        snippet_text = "\n\n".join(selections)
        identifiers = _extract_identifiers(snippet_text)
        if not identifiers:
            identifiers = {""}  # avoid zero-division later

        origin_dir = os.path.dirname(self.view.file_name() or "")

        # 2. Build candidate set ---------------------------------------------
        candidate_paths: Set[str] = set()
        for ident in identifiers:
            for path, *_ in window.lookup_symbol_in_index(ident):
                candidate_paths.add(path)

        scored: List[Tuple[str, float, str]] = []  # (path, score, contents)
        for path in candidate_paths:
            if not os.path.isfile(path):
                continue
            try:
                txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            distinct_hits = sum(
                1 for ident in identifiers if re.search(rf"\b{re.escape(ident)}\b", txt)
            )
            total_hits = sum(
                len(re.findall(rf"\b{re.escape(ident)}\b", txt))
                for ident in identifiers
            )
            size_kb = len(txt) / 1024
            same_dir = os.path.dirname(path) == origin_dir

            score = (
                100 * distinct_hits + 2 * total_hits + (20 if same_dir else 0) - size_kb
            )
            scored.append((path, score, txt))

        scored.sort(key=lambda t: -t[1])
        if not scored:
            sublime.status_message(
                "LLM-Prompt: no related files; generating prompt anyway…"
            )
            self._finalise_prompt(window, snippet_text, [], identifiers)
            return

        # 3. Interactive picker (top-5 pre-selected, sticky index) ------------
        included = [(i < 5) for i in range(len(scored))]

        def _display_items() -> List[str]:
            root = window.folders()[0] if window.folders() else ""
            return [
                ("☑" if inc else "☐") + " " + os.path.relpath(p, root)
                for (p, _, _), inc in zip(scored, included)
            ] + ["⏎ Done / Generate"]

        def _on_done(idx: int) -> None:
            if idx == -1:
                return
            if idx == len(included):  # Done
                chosen = [(p, txt) for (p, _, txt), inc in zip(scored, included) if inc]
                self._finalise_prompt(window, snippet_text, chosen, identifiers)
                return
            included[idx] = not included[idx]
            window.show_quick_panel(_display_items(), _on_done, _MONO, idx)

        window.show_quick_panel(_display_items(), _on_done, _MONO)

    # ---------------------------------------------------------------------
    #  Prompt composer
    # ---------------------------------------------------------------------

    def _finalise_prompt(
        self,
        window: sublime.Window,
        snippet: str,
        related: List[Tuple[str, str]],
        idents: Set[str],
    ) -> None:
        project_root = (window.folders() or ["."])[0]

        # Overview -----------------------------------------------------------
        overview_lines: List[str] = []
        for path, txt in related:
            lang = _syntax_from_extension(path)
            defs = _extract_definitions(txt, lang)
            if not defs:
                continue
            rel = os.path.relpath(path, project_root)
            for cls, methods in defs.items():
                if cls and cls not in idents:
                    continue
                if cls:
                    overview_lines.append(f"- class {cls}  ←  {rel}")
                    for name, sig in methods:
                        overview_lines.append(f"  • {name}{sig}")
                else:
                    for name, sig in methods:
                        if name in idents:
                            overview_lines.append(f"- function {name}{sig}  ←  {rel}")
        overview_text = "\n".join(overview_lines)

        # Compose prompt -----------------------------------------------------
        parts: List[str] = []
        parts.append("<user_question>\n# Your question\n\n</user_question>\n\n")

        parts.append("<file_map>\n")
        for i, (path, _) in enumerate(related, 1):
            parts.append(f"{i}. {os.path.relpath(path, project_root)}\n")
        parts.append("</file_map>\n\n")

        sel_lang = self.view.syntax().name.lower() if self.view.syntax() else ""
        parts.append(
            f'<file_contents path="<selection>">\n```{sel_lang}\n{snippet}\n```\n</file_contents>\n\n'
        )

        for path, txt in related:
            lang = _syntax_from_extension(path)
            rel = os.path.relpath(path, project_root)
            parts.append(
                f'<file_contents path="{rel}">\n```{lang}\n{txt}\n```\n</file_contents>\n\n'
            )

        if overview_text:
            parts.append("<overview>\n")
            parts.append(textwrap.dedent(overview_text))
            parts.append("\n</overview>\n")

        prompt_text = "".join(parts)

        # Spawn prompt tab ---------------------------------------------------
        prompt_view = window.new_file()
        prompt_view.set_scratch(True)
        prompt_view.set_name("LLM-Prompt.md")
        prompt_view.assign_syntax("Packages/Markdown/Markdown.sublime-syntax")
        prompt_view.run_command("append", {"characters": prompt_text})

        q_start = prompt_text.find("# Your question") + len("# Your question\n\n")
        prompt_view.sel().clear()
        prompt_view.sel().add(sublime.Region(q_start, q_start))
        prompt_view.show_at_center(q_start)

        window.settings().set(_SETTINGS_KEY, prompt_view.id())
        sublime.status_message("LLM-Prompt: prompt generated ✨")


# ---------------------------------------------------------------------------
#  Add selection to existing prompt
# ---------------------------------------------------------------------------


class AddSelectionToPromptCommand(sublime_plugin.TextCommand):
    def run(self, edit: sublime.Edit) -> None:  # noqa: D401
        window = self.view.window()
        if window is None:
            return
        prompt_id = window.settings().get(_SETTINGS_KEY)
        prompt_view = next((v for v in window.views() if v.id() == prompt_id), None)
        if prompt_view is None:
            sublime.status_message("LLM-Prompt: no active prompt.")
            return

        selections = [self.view.substr(r) for r in self.view.sel() if not r.empty()]
        if not selections:
            sublime.status_message("LLM-Prompt: nothing selected.")
            return

        sel_text = "\n\n".join(selections)
        lang = self.view.syntax().name.lower() if self.view.syntax() else ""
        snippet = (
            f'\n<file_contents path="<additional-selection>">\n'
            f"```{lang}\n{sel_text}\n```\n</file_contents>\n"
        )
        prompt_view.run_command("append", {"characters": snippet})
        sublime.status_message("LLM-Prompt: context appended ✅")
