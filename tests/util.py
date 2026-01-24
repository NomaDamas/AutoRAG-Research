from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Result of a single verification check."""

    passed: bool
    message: str = ""
    failures: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Aggregated report of all verification checks."""

    checks: dict[str, CheckResult] = field(default_factory=dict)

    def add_check(self, name: str, result: CheckResult) -> None:
        """Add a check result to the report."""
        self.checks[name] = result

    @property
    def all_passed(self) -> bool:
        """Return True if all checks passed."""
        return all(check.passed for check in self.checks.values())

    def summary(self, title: str = "Verification Report") -> str:
        """Return a summary of all checks."""
        lines = [f"=== {title} ==="]
        for name, result in self.checks.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{status}] {name}: {result.message}")
            for failure in result.failures:
                lines.append(f"    - {failure}")
        return "\n".join(lines)
