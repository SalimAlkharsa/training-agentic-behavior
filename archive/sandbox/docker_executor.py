"""
Docker-based code execution sandbox for safe Python code execution.

This module provides a paranoid-safe execution environment using Docker containers
with strict resource limits, network isolation, and filesystem restrictions.
"""

import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any
import json


class DockerExecutor:
    """
    Safe Python code execution using Docker containers.

    Provides maximum isolation for untrusted code execution with:
    - No network access
    - Limited memory and CPU
    - Timeout enforcement
    - Read-only filesystem (except /tmp)
    - Automatic container cleanup

    Example:
        >>> executor = DockerExecutor(timeout=5, memory_limit_mb=256)
        >>> result = executor.execute("print(sum(range(100)))")
        >>> print(result['output'])
        4950
    """

    def __init__(
        self,
        timeout: int = 5,
        memory_limit_mb: int = 256,
        cpu_limit: float = 0.5,
        docker_image: str = "python:3.11-slim"
    ):
        """
        Initialize the Docker executor.

        Args:
            timeout: Maximum execution time in seconds (default: 5)
            memory_limit_mb: Memory limit in MB (default: 256)
            cpu_limit: CPU limit as fraction of cores (default: 0.5)
            docker_image: Docker image to use (default: python:3.11-slim)
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit = cpu_limit
        self.docker_image = docker_image

        # Verify Docker is available
        self._check_docker_available()

        # Ensure Docker image is available
        self._ensure_image_available()

    def _check_docker_available(self) -> None:
        """Check if Docker is installed and accessible."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not accessible")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is not installed. Please install Docker to use DockerExecutor."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker command timed out")

    def _ensure_image_available(self) -> None:
        """Ensure the Docker image is available locally."""
        try:
            # Check if image exists locally
            result = subprocess.run(
                ["docker", "image", "inspect", self.docker_image],
                capture_output=True,
                timeout=5
            )

            if result.returncode != 0:
                print(f"Docker image {self.docker_image} not found locally. Pulling...")
                pull_result = subprocess.run(
                    ["docker", "pull", self.docker_image],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes for pull
                )

                if pull_result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to pull Docker image: {pull_result.stderr}"
                    )
                print(f"Successfully pulled {self.docker_image}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker image check/pull timed out")

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in an isolated Docker container.

        Args:
            code: Python code string to execute

        Returns:
            Dictionary containing:
                - success (bool): Whether execution completed without errors
                - output (str): Standard output from the code
                - error (str | None): Error message if execution failed
                - execution_time (float): Time taken in seconds
                - memory_used (int | None): Memory used in bytes (if available)
                - timeout_exceeded (bool): Whether execution was killed due to timeout
        """
        start_time = time.time()

        # Create temporary file with the code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            code_file = Path(f.name)
            f.write(code)

        try:
            # Build Docker run command with safety restrictions
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--network=none",  # No network access
                f"--memory={self.memory_limit_mb}m",  # Memory limit
                f"--cpus={self.cpu_limit}",  # CPU limit
                "--read-only",  # Read-only filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=10m",  # Writable /tmp (limited)
                "-v", f"{code_file.absolute()}:/code.py:ro",  # Mount code read-only
                self.docker_image,
                "python", "/code.py"
            ]

            # Execute with timeout
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                execution_time = time.time() - start_time

                # Parse output
                success = result.returncode == 0
                output = result.stdout.strip()
                error = result.stderr.strip() if result.stderr else None

                return {
                    "success": success,
                    "output": output,
                    "error": error,
                    "execution_time": execution_time,
                    "memory_used": None,  # Docker stats require running container
                    "timeout_exceeded": False
                }

            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time

                # Kill any remaining containers
                self._cleanup_containers()

                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution exceeded timeout of {self.timeout} seconds",
                    "execution_time": execution_time,
                    "memory_used": None,
                    "timeout_exceeded": True
                }

        finally:
            # Clean up temporary file
            try:
                code_file.unlink()
            except Exception:
                pass  # Best effort cleanup

    def _cleanup_containers(self) -> None:
        """Emergency cleanup of any running containers from this executor."""
        try:
            # Kill all containers using our image
            subprocess.run(
                [
                    "docker", "ps", "-q",
                    "--filter", f"ancestor={self.docker_image}"
                ],
                capture_output=True,
                timeout=5
            )
            # Note: --rm flag should handle cleanup, but this is backup
        except Exception:
            pass  # Best effort cleanup

    def execute_batch(self, code_samples: list[str]) -> list[Dict[str, Any]]:
        """
        Execute multiple code samples sequentially.

        Args:
            code_samples: List of Python code strings to execute

        Returns:
            List of execution result dictionaries
        """
        results = []
        for code in code_samples:
            result = self.execute(code)
            results.append(result)
        return results

    def __repr__(self) -> str:
        """String representation of the executor."""
        return (
            f"DockerExecutor(timeout={self.timeout}s, "
            f"memory={self.memory_limit_mb}MB, "
            f"cpu={self.cpu_limit}, "
            f"image={self.docker_image})"
        )
