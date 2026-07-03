# devrun heartbeat

Heartbeat scheduler control — manage the background service or run the scheduler loop in the foreground.

## Synopsis

```bash
devrun heartbeat                    # Run foreground loop (refuses if service active)
devrun heartbeat run                # Run foreground without service guard
devrun heartbeat install            # Install service (does not start)
devrun heartbeat start              # Activate the installed service
devrun heartbeat start              # Start service
devrun heartbeat stop               # Stop service
devrun heartbeat restart            # Restart service
devrun heartbeat status             # Show service status and job counts
devrun heartbeat uninstall          # Remove service
```

## Description

The heartbeat scheduler is the core orchestration engine for dependency-aware workflows. It runs a periodic tick loop that:

1. **Reclaims stale leases** from crashed or hung scheduler instances
2. **Expires workflow deadlines** for workflows that exceed their configured timeout
3. **Cascade-skips dependents** of failed or skipped jobs
4. **Promotes ready QUEUED jobs** to SUBMITTED when all dependencies are satisfied
5. **Polls active jobs** (SUBMITTED/RUNNING/CANCELING) and updates their status

The heartbeat can run as a managed background service (via systemd on Linux or launchd on macOS) or in the foreground for debugging and development.

---

## Commands

### `devrun heartbeat` (default)

Run the heartbeat loop in the foreground. This is the default command when no subcommand is specified.

**Behavior:**
- Checks if the background service is active
- If the service is running, refuses to start and exits with code 1
- Otherwise, starts the tick loop with a 10-second interval
- Writes tick timestamps to `~/.devrun/heartbeat.tick`
- Handles SIGTERM and SIGINT for clean shutdown

**Use when:**
- Testing heartbeat behavior during development
- Running workflows without installing a persistent service

**Exit codes:**
- `0` — Clean shutdown (SIGTERM/SIGINT received)
- `1` — Background service is already running

**Example:**

```bash
# Run in foreground (Ctrl+C to stop)
devrun heartbeat
```

---

### `devrun heartbeat run`

Run the foreground loop without checking whether the service is active.

**Behavior:**
- Identical to `devrun heartbeat` but skips the service-active guard
- Useful for manual supervision or running under a custom process manager

**Use when:**
- You want explicit control over the scheduler process
- Running under a supervisor like `supervisord` or Docker

**Example:**

```bash
# Run without service check
devrun heartbeat run
```

---

### `devrun heartbeat install`

Install the heartbeat as a managed service. **Does NOT start it**; run `devrun heartbeat start` to activate.

**Behavior:**
- Detects the host platform (Linux → systemd, macOS → launchd)
- Generates a service unit file with the current Python interpreter path and default DB location
- Registers the service with the OS service manager
- Enables auto-start on boot/login
- Leaves the service stopped — run `devrun heartbeat start` to activate

**Service file locations:**
- **Linux:** `~/.config/systemd/user/devrun-heartbeat.service`
- **macOS:** `~/Library/LaunchAgents/com.devrun.heartbeat.plist`

**Exit codes:**
- `0` — Service installed successfully (not started)
- Non-zero — Installation failed (systemctl/launchctl error)

**Example:**

```bash
# Install the service (does not start)
devrun heartbeat install

# Start the service
devrun heartbeat start

# Verify it's running
devrun heartbeat status
```

---

### `devrun heartbeat start`

Start the installed heartbeat service.

**Behavior:**
- Starts the service via `systemctl --user start` (Linux) or `launchctl kickstart` (macOS)
- Requires the service to be installed first (`devrun heartbeat install`)

**Exit codes:**
- `0` — Service started successfully
- Non-zero — Service not installed or start failed

**Example:**

```bash
# Start the service
devrun heartbeat start
```

---

### `devrun heartbeat stop`

Stop the running heartbeat service.

**Behavior:**
- Stops the service via `systemctl --user stop` (Linux) or `launchctl stop` (macOS)
- The service remains installed and can be restarted

**Exit codes:**
- `0` — Service stopped successfully
- Non-zero — Stop command failed

**Example:**

```bash
# Stop the service
devrun heartbeat stop
```

---

### `devrun heartbeat restart`

Restart the heartbeat service.

**Behavior:**
- Stops and starts the service in one operation
- Equivalent to `stop` followed by `start`

**Exit codes:**
- `0` — Service restarted successfully
- Non-zero — Restart failed

**Example:**

```bash
# Restart the service (e.g., after config changes)
devrun heartbeat restart
```

---

### `devrun heartbeat status`

Display service status, job status counts, and last tick timestamp.

**Output:**
- **Service status:** `active` or `inactive`
- **Job status counts:** Number of jobs in each state (QUEUED, SUBMITTED, RUNNING, COMPLETED, FAILED, etc.)
- **Last tick timestamp:** ISO 8601 timestamp of the most recent heartbeat tick

**Exit codes:**
- `0` — Status retrieved successfully

**Example:**

```bash
devrun heartbeat status
```

**Sample output:**

```
Service: active
Job status counts:
  COMPLETED: 42
  FAILED: 3
  QUEUED: 5
  RUNNING: 2
  SUBMITTED: 1
Last tick: 2026-06-13T14:32:10.123456+00:00
```

---

### `devrun heartbeat uninstall`

Remove the heartbeat service from the system.

**Behavior:**
- Stops the service if running
- Disables auto-start
- Removes the service unit file
- Reloads the service manager daemon

**Exit codes:**
- `0` — Service uninstalled successfully
- Non-zero — Uninstall failed

**Example:**

```bash
# Remove the service completely
devrun heartbeat uninstall
```

---

## Environment

The heartbeat scheduler uses the following paths:

| Path | Purpose |
|------|---------|
| `~/.devrun/jobs.db` | Default SQLite database location |
| `~/.devrun/heartbeat.tick` | Timestamp of the last tick (foreground mode) |
| `~/.config/systemd/user/devrun-heartbeat.service` | systemd unit file (Linux) |
| `~/Library/LaunchAgents/com.devrun.heartbeat.plist` | launchd plist (macOS) |

---

## Platform Support

| Platform | Service Manager | Auto-start | Status Command |
|----------|----------------|------------|----------------|
| Linux | systemd (`--user`) | Yes (login) | `systemctl --user status devrun-heartbeat` |
| macOS | launchd | Yes (login) | `launchctl print gui/$(id -u)/com.devrun.heartbeat` |

---

## See Also

- [Heartbeat Service Installation Guide](../guides/heartbeat-service.md)
- `devrun workflow run --help` — Submit dependency-aware workflows
- `devrun status` — Check job status
