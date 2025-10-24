from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.sandbox import CommandResult, Permission, Sandbox, SandboxError


def test_fs_write_read_with_permissions(tmp_path: Path) -> None:
    events = []
    sandbox = Sandbox(tmp_path / 'ws', permissions={Permission.READ, Permission.WRITE}, audit_logger=events.append)
    sandbox.fs_write('docs/hello.txt', 'hello world')
    content = sandbox.fs_read('docs/hello.txt')
    assert content == 'hello world'
    paths = sandbox.fs_list('docs')
    assert any(path.name == 'hello.txt' for path in paths)
    assert any(event['action'] == 'fs_write' for event in events)


def test_fs_write_without_permission_raises(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / 'ws', permissions={Permission.READ})
    with pytest.raises(SandboxError):
        sandbox.fs_write('docs/hello.txt', 'hello world')


def test_path_escape_disallowed(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / 'ws', permissions={Permission.WRITE})
    with pytest.raises(SandboxError):
        sandbox.fs_write('../outside.txt', 'nope')


def test_net_download_with_mock(tmp_path: Path) -> None:
    events = []
    sandbox = Sandbox(tmp_path / 'ws', permissions={Permission.READ, Permission.WRITE, Permission.NET}, audit_logger=events.append)

    fake_response = Mock()
    fake_response.content = b'data'
    fake_response.raise_for_status = Mock()

    with patch('core.sandbox.requests.get', return_value=fake_response) as mock_get:
        result = sandbox.net_download('https://example.com/file.txt', 'downloads/file.txt')

    mock_get.assert_called_once()
    assert result['sha256']
    assert (tmp_path / 'ws' / 'downloads' / 'file.txt').exists()
    assert any(event['action'] == 'net_download' for event in events)


def test_run_command_requires_exec_permission(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / 'ws', permissions=set())
    with pytest.raises(SandboxError):
        sandbox.run_command(['python', '--version'])

    sandbox.permissions.add(Permission.EXEC)
    result = sandbox.run_command(['python', '--version'])
    assert isinstance(result, CommandResult)
    assert result.returncode == 0
