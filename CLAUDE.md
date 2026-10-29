@AGENTS.md

# Reporting

For every final report to the user, follow ASD-STE100 Simplified Technical English principles and translate the final report into Korean.

- Use short, direct sentences.
- Use approved ASD-STE100 vocabulary principles where possible, and translate clearly into Korean.
- Use one meaning for each word or term in a report.
- State the result first. Then state changed files and verification.
- Do not use idioms, figurative language, or unnecessary filler.
- Do not claim formal ASD-STE100 compliance unless the text is validated with an approved checker.

# Precise technical explanations

For every technical explanation, use labels that match the actual event and state.
Distinguish initialization, prediction, measurement, optimization, and publication.
In a sequence, identify the anchor and target by frame or time. Do not call a
prediction for the frame after the anchor the first-frame state. If a short
label is ambiguous or unverified, use a precise label or state the uncertainty.
Apply this rule in all sessions. For detailed time-indexed explanations, use
the `precise-state-explanations` skill (`.claude/skills/precise-state-explanations`).
