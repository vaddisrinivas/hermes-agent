import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _RetryableFailureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        self._set_fatal_error(
            "telegram_connect_error",
            "Telegram startup failed: temporary DNS resolution failure.",
            retryable=True,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _DisabledAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=False, token="***"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        raise AssertionError("connect should not be called for disabled platforms")

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SuccessfulAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_returns_failure_for_retryable_startup_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _RetryableFailureAdapter())

    ok = await runner.start()

    assert ok is False
    assert runner.should_exit_cleanly is False
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    assert "temporary DNS resolution failure" in state["exit_reason"]
    assert state["platforms"]["telegram"]["state"] == "retrying"
    assert state["platforms"]["telegram"]["error_code"] == "telegram_connect_error"


@pytest.mark.asyncio
async def test_runner_allows_cron_only_mode_when_no_platforms_are_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.adapters == {}
    state = read_runtime_status()
    assert state["gateway_state"] == "running"


@pytest.mark.asyncio
async def test_runner_records_connected_platform_state_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _SuccessfulAdapter())
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    ok = await runner.start()

    assert ok is True
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["platforms"]["discord"]["state"] == "connected"
    assert state["platforms"]["discord"]["error_code"] is None
    assert state["platforms"]["discord"]["error_message"] is None


@pytest.mark.asyncio
async def test_start_gateway_verbosity_imports_redacting_formatter(monkeypatch, tmp_path):
    """Verbosity != None must not crash with NameError on RedactingFormatter (#8044)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    # verbosity=1 triggers the code path that uses RedactingFormatter.
    # Before the fix this raised NameError.
    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=1)

    assert ok is True


@pytest.mark.asyncio
async def test_start_gateway_replace_force_uses_terminate_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls = []

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda: 0)
    monkeypatch.setattr("gateway.status.terminate_pid", lambda pid, force=False: calls.append((pid, force)))
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("gateway.run.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    assert calls == [(42, False), (42, True)]


class _CustomTestAdapter(BasePlatformAdapter):
    """Mock adapter for testing platform_class dynamic loading."""

    def __init__(self, config):
        super().__init__(config, Platform.TELEGRAM)
        self.config_received = config

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class TestPlatformClassConfig:
    """Tests for platform_class field allowing custom adapter loading."""

    def test_platform_class_in_config_roundtrip(self):
        """platform_class should persist through to_dict/from_dict."""
        pc = PlatformConfig(
            enabled=True,
            token="test-token",
            platform_class="my_custom.adapters.MyAdapter",
        )
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)

        assert restored.platform_class == "my_custom.adapters.MyAdapter"

    def test_platform_class_none_by_default(self):
        """platform_class should be None when not set."""
        pc = PlatformConfig(enabled=True, token="test-token")
        assert pc.platform_class is None

        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)
        assert restored.platform_class is None

    def test_create_adapter_uses_platform_class_when_set(self, tmp_path, monkeypatch):
        """_create_adapter should use dynamic import when platform_class is set."""
        import sys

        # Create a mock adapter module
        mock_adapter_code = '''
class MockCustomAdapter:
    def __init__(self, config):
        self.config = config
        self.platform = "mock_platform"

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}
'''

        # Create a temporary module
        import types
        mock_module = types.ModuleType("_mock_custom_adapter")
        exec(mock_adapter_code, mock_module.__dict__)
        sys.modules["_mock_custom_adapter"] = mock_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        pc = PlatformConfig(
            enabled=True,
            token="test-token",
            platform_class="_mock_custom_adapter.MockCustomAdapter",
        )

        config = GatewayConfig(
            platforms={Platform.TELEGRAM: pc},
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)

        # Create adapter using the platform_class
        adapter = runner._create_adapter(Platform.TELEGRAM, pc)

        assert adapter is not None
        assert type(adapter).__name__ == "MockCustomAdapter"
        assert adapter.config is pc

        # Cleanup
        del sys.modules["_mock_custom_adapter"]

    def test_create_adapter_returns_none_on_invalid_platform_class(self, tmp_path, monkeypatch, caplog):
        """Invalid platform_class should log error and return None."""
        import logging

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        pc = PlatformConfig(
            enabled=True,
            token="test-token",
            platform_class="nonexistent.module.BadAdapter",
        )

        config = GatewayConfig(
            platforms={Platform.TELEGRAM: pc},
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)

        adapter = runner._create_adapter(Platform.TELEGRAM, pc)

        assert adapter is None
        # Should log an error about failed import
        assert any("Failed to load custom platform adapter" in record.message
                   for record in caplog.records if record.levelno == logging.ERROR)
