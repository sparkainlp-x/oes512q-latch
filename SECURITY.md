# Security Policy

Please report potential vulnerabilities privately to the repository owner rather than opening a public issue. Include a minimal reproduction, affected version or commit, and the security impact. Avoid publishing exploit details until a remediation path has been agreed.

Invalid input (wrong length, non-finite or non-real values, invalid thresholds) is rejected with `ValueError` / `TypeError` by design (fail closed); a case where invalid input is silently admitted is in scope.
