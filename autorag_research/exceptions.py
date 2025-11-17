class EnvNotFoundError(Exception):
    """Raised when a required environment variable is not found."""

    def __init__(self, env_var_name: str):
        super().__init__(f"Environment variable '{env_var_name}' not found.")


class NoSessionError(Exception):
    """Raised when there is no active database session."""

    def __init__(self):
        super().__init__("No active database session found.")
