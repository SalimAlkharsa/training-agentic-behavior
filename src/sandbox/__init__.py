"""
Safe code execution sandbox module.

Provides Docker-based isolated execution environment for untrusted Python code.
"""

from .docker_executor import DockerExecutor

__all__ = ["DockerExecutor"]
