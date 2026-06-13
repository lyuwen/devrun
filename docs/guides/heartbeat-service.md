# Heartbeat Service Installation Guide

The devrun heartbeat scheduler is a background service that orchestrates dependency-aware workflows. It continuously monitors queued jobs, promotes them when dependencies are satisfied, polls running jobs for status updates, and handles failure propagation.

## What It Does

The heartbeat runs a periodic **tick loop** (default: 10 seconds) that executes five phases in order:

1. **Reclaim stale leases** — Recover jobs from crashed scheduler instances
2. **Expire workflow deadlines** — Mark timed-out workflows as failed
3. **Cascade-skip dependents** — Skip jobs that depend on failed parents
4. **Promote ready QUEUED jobs** — Submit jobs whose dependencies are satisfied
5. **Poll active jobs** — Update status for SUBMITTED/RUNNING/CANCELING jobs

This stateful orchestration enables workflows to span hours or days without manual intervention. Jobs with `depends_on` relationships are automatically sequenced, and failures propagate to downstream dependents.

## Platform Support

The heartbeat service integrates with native OS service managers:

- **Linux:** systemd user services (`systemctl --user`)
- **macOS:** launchd agents (`launchctl`)

Both backends support auto-start on login and graceful shutdown via `SIGTERM`.

## Installation

Install and start the service in one command:

```bash
devrun heartbeat install
```

This command:
- Detects your platform (Linux or macOS)
- Generates a service configuration file with your current Python interpreter and default database path
- Registers the service with the OS service manager
- Enables auto-start on boot/login
- Starts the service immediately

**Service file locations:**

| Platform | Path |
|----------|------|
| Linux | `~/.config/systemd/user/devrun-heartbeat.service` |
| macOS | `~/Library/LaunchAgents/com.devrun.heartbeat.plist` |

## Verification

Check that the service is running:

```bash
devrun heartbeat status
```

**Expected output:**

```
Service: active
Job status counts:
  QUEUED: 0
  SUBMITTED: 0
  RUNNING: 0
  COMPLETED: 0
Last tick: 2026-06-13T14:32:10.123456+00:00
```

If the service is `inactive`, check the logs (see below).

## Managing the Service

Once installed, use these commands to control the service:

```bash
# Start the service
devrun heartbeat start

# Stop the service
devrun heartbeat stop

# Restart the service (e.g., after configuration changes)
devrun heartbeat restart

# Remove the service completely
devrun heartbeat uninstall
```

## Viewing Logs

### Linux (systemd)

View live logs:

```bash
journalctl --user -u devrun-heartbeat.service -f
```

View recent logs:

```bash
journalctl --user -u devrun-heartbeat.service -n 50
```

### macOS (launchd)

Logs are written to system log files viewable in **Console.app**:

1. Open Console.app
2. Search for `devrun.heartbeat` in the search bar
3. Filter by process name: `python`

Alternatively, use `log` command:

```bash
log show --predicate 'processImagePath contains "python"' --info --last 10m | grep heartbeat
```

## Troubleshooting

### Service fails to start

**Symptom:** `devrun heartbeat status` shows `inactive` immediately after `install`.

**Common causes:**

1. **Missing executors.yaml**

   The heartbeat requires `~/.devrun/executors.yaml` to resolve executor configurations.

   **Fix:** Create the file with at least a `local` executor:

   ```yaml
   local:
     type: local
   ```

2. **Incorrect database path**

   The service is hardcoded to use `~/.devrun/jobs.db` at install time. If you've moved the database, the service won't find it.

   **Fix:** Uninstall and reinstall the service, or manually edit the service file to update the `--db` path.

3. **Python environment issues**

   The service uses the Python interpreter active at `install` time. If that environment is deleted or moved, the service will fail.

   **Fix:** Reinstall the service from the correct Python environment.

### Jobs stay in QUEUED state

**Symptom:** `devrun status` shows jobs stuck in `QUEUED` status indefinitely.

**Possible causes:**

1. **Service not running**

   Check `devrun heartbeat status`. If `inactive`, start it with `devrun heartbeat start`.

2. **Dependencies not satisfied**

   Jobs with `depends_on` relationships wait for parent jobs to complete. Check parent job status with `devrun status`.

3. **Unfilled `<REQUIRED:...>` placeholders**

   The heartbeat validates job parameters before submission. Jobs with unresolved placeholders are marked `SKIPPED` with a skip reason.

   **Fix:** Check `devrun status <job_id>` for the skip reason, then provide missing parameters.

### Service stops unexpectedly

**Symptom:** Service was `active` but is now `inactive` without manual intervention.

**Possible causes:**

1. **Unhandled exception in tick loop**

   Check logs for Python tracebacks. The service restarts on failure (Linux: `Restart=on-failure`, macOS: `KeepAlive=true`), but repeated crashes may indicate a configuration issue.

2. **Database locked**

   SQLite databases are not designed for high-concurrency writes. If multiple processes try to write simultaneously, one may fail with `OperationalError: database is locked`.

   **Fix:** Ensure only one heartbeat instance is running. Check for stray processes with `ps aux | grep devrun.heartbeat`.

## Advanced Configuration

### Custom tick interval

The default tick interval is 10 seconds. To change it, manually edit the service file:

**Linux (`~/.config/systemd/user/devrun-heartbeat.service`):**

```ini
ExecStart=/path/to/python -m devrun.heartbeat --db /path/to/jobs.db --interval 5.0
```

**macOS (`~/Library/LaunchAgents/com.devrun.heartbeat.plist`):**

```xml
<key>ProgramArguments</key>
<array>
  <string>/path/to/python</string>
  <string>-m</string>
  <string>devrun.heartbeat</string>
  <string>--db</string>
  <string>/path/to/jobs.db</string>
  <string>--interval</string>
  <string>5.0</string>
</array>
```

After editing, reload and restart:

```bash
# Linux
systemctl --user daemon-reload
devrun heartbeat restart

# macOS
launchctl unload ~/Library/LaunchAgents/com.devrun.heartbeat.plist
launchctl load ~/Library/LaunchAgents/com.devrun.heartbeat.plist
```

### Running without a service (foreground mode)

For development or debugging, run the heartbeat in the foreground:

```bash
devrun heartbeat
```

This runs the tick loop in the current terminal. Press `Ctrl+C` to stop cleanly. The foreground command refuses to start if the background service is active (use `devrun heartbeat stop` first).

## See Also

- [Heartbeat CLI Reference](../cli/heartbeat.md) — Full command documentation
- `devrun workflow run --help` — Submit dependency-aware workflows
- `devrun status` — Check job and workflow status
