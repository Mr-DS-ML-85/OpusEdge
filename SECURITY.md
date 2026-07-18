# Security Policy

## Supported versions

OpusEdge is pre-1.0 research software. The latest tagged release always
receives fixes; older tags are best-effort.

## Reporting a vulnerability

If you find a security issue — memory corruption, ReDoS in the Python bench,
supply-chain risk, anything that could compromise a user running this code —
**please do NOT open a public issue**.

Use one of:

- GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repo (the "Report a vulnerability" button on the Security tab).
- Email the maintainer address listed on the GitHub profile.

Please include:

- A short description of the issue and its impact.
- Reproduction steps (a small program or bench invocation is ideal).
- Any suggested fix, if you have one.

## Response timeline

Best-effort, Bangladesh-time, mostly async:

- **72 hours** — acknowledgement.
- **7 days** — triage + severity assignment.
- **30 days** — fix or public advisory, whichever is safer.

We coordinate disclosure — please give us the fix window before publishing.

## Scope

In scope: the C++ engine (`engine/`), the Python bindings, the bench harness
(`bench/`), the SDK (`opusedge_cpp.sdk`).

Out of scope: the landing page (`web/`) beyond obvious XSS, third-party
dependencies (report those upstream), and issues that require an attacker to
already control the machine running the bench.
