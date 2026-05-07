# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sourced from https://github.com/vllm-project/vllm/blob/main/docs/mkdocs/hooks/url_schemes.py"""  # noqa: E501

import regex as re
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files: Files):
    gh_icon = ":octicons-mark-github-16:"
    gh_url = "https://github.com"
    repo_url = f"{gh_url}/torch-spyre/spyre-inference"
    org_url = f"{gh_url}/orgs/vllm-project"
    urls = {
        "issue": f"{repo_url}/issues",
        "pr": f"{repo_url}/pull",
        "project": f"{org_url}/projects",
        "dir": f"{repo_url}/tree/main",
        "file": f"{repo_url}/blob/main",
    }
    titles = {
        "issue": "Issue #",
        "pr": "Pull Request #",
        "project": "Project #",
        "dir": "",
        "file": "",
    }

    scheme = r"gh-(?P<type>.+?):(?P<path>.+?)(#(?P<fragment>.+?))?"
    inline_link = re.compile(r"\[(?P<title>[^\[]+?)\]\(" + scheme + r"\)")
    auto_link = re.compile(f"<{scheme}>")

    def replace_inline_link(match: re.Match) -> str:
        url = f"{urls[match.group('type')]}/{match.group('path')}"
        if fragment := match.group("fragment"):
            url += f"#{fragment}"

        return f"[{gh_icon} {match.group('title')}]({url})"

    def replace_auto_link(match: re.Match) -> str:
        type = match.group("type")
        path = match.group("path")
        title = f"{titles[type]}{path}"
        url = f"{urls[type]}/{path}"
        if fragment := match.group("fragment"):
            url += f"#{fragment}"

        return f"[{gh_icon} {title}]({url})"

    markdown = inline_link.sub(replace_inline_link, markdown)
    markdown = auto_link.sub(replace_auto_link, markdown)

    return markdown
