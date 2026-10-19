# Third-Party Notices

LiteScale contains a mix of original code and code derived from upstream open-source projects.
This file is an attribution and provenance summary for the main upstream code families that
materially affect repository licensing.

## Repository licensing model

- Unless otherwise noted in file headers, original LiteScale code is licensed under Apache-2.0.
- Files copied from or derived from upstream projects remain subject to their original copyright
  notices and license terms.
- When a file was modified after being imported from an upstream project, the upstream notice must
  be retained and the modification should be identified in the file header or commit history.

## Major upstream sources

### Megatron-LM

- Upstream project: NVIDIA Megatron-LM
- Upstream repository: https://github.com/NVIDIA/Megatron-LM
- Upstream license: BSD-3-Clause style NVIDIA license
- Relevant local areas: `megatron/`, `pretrain_gpt.py`, and any files that retain Megatron-LM
  copyright or provenance notices
- Redistribution notes:
  - retain the original copyright notice
  - retain the BSD disclaimer text
  - do not use NVIDIA or contributor names to endorse derived products without permission

### PAI-Megatron-Patch

- Upstream project: PAI-Megatron-Patch
- Upstream repository: https://github.com/alibaba/Pai-Megatron-Patch
- Upstream license: Apache License 2.0
- Relevant local areas: files adapted from PAI-Megatron-Patch in LiteScale training, launcher,
  and distributed execution flows, where applicable
- Redistribution notes:
  - retain original copyright and license notices
  - mark modified files prominently
  - include the Apache-2.0 license text with redistribution

## Other third-party code families already present in the codebase

This repository may also contain files or fragments derived from other upstream projects, including
projects already referenced by Megatron-LM or other imported components. Where those files retain
their original headers, those notices continue to govern the applicable file-level obligations.

Examples may include code families under Apache-2.0, MIT, or BSD-style licenses. If additional
upstream imports are added later, this file should be updated together with the corresponding file
headers.

## Practical release guidance

- Keep file-level upstream notices intact.
- Do not replace upstream file headers with a single repository-wide notice.
- If you add new copied or adapted files from Megatron-LM or PAI-Megatron-Patch, preserve their
  original notices in those files.
- If you substantially refactor an imported file, keep the upstream attribution and add a local
  "Modified by" note.