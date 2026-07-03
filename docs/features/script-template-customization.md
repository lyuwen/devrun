# Script Template Customization

## Overview

The `SWEBenchAgenticTask` supports flexible customization of the executed Python script and its CLI arguments through three mechanisms:

1. **Parameter-based customization** — via `run_script` and `script_args` parameters
2. **Hook method override** — via `_get_script_args()` subclass method
3. **Custom template** — via `template` parameter for complete control

## Usage Examples

### Example 1: Simple Script and Args Override (YAML Config)

```yaml
task: swe_bench_agentic
executor: slurm_swedev2

params:
  # Custom script path
  run_script: "custom_scripts/my_inference.py"
  
  # Additional CLI arguments
  script_args:
    batch_size: 32
    temperature: 0.7
    enable_cache: true
    disable_logging: false  # false values are omitted
  
  # Standard params
  dataset: "/data/swe_bench"
  array: "000-099"
  llm_config: "/configs/model.json"
```

Generated command excerpt:
```bash
python custom_scripts/my_inference.py "${_LLM_CONFIG}" \
    --dataset ${DATASET} \
    --split test \
    --max-iterations 50 \
    --max-attempts 5 \
    --select /data/select/000.txt \
    --workspace /tmp/workspace \
    --output-dir /output/000 \
    --use-legacy-tools \
    --bind-dev-sdk \
    --batch-size 32 \
    --temperature 0.7 \
    --enable-cache
```

### Example 2: Task Inheritance for Complex Customization

```python
# devrun/tasks/my_custom_eval.py
from typing import Any
from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask
from devrun.registry import register_task

@register_task("my_custom_eval")
class MyCustomEvalTask(SWEBenchAgenticTask):
    """Custom evaluation task with domain-specific arguments."""
    
    def _get_run_script(self, params: dict[str, Any]) -> str:
        return "my_tools/custom_evaluator.py"
    
    def _get_script_args(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "evaluator_mode": params.get("mode", "strict"),
            "enable_cache": True,
            "metrics": ",".join(params.get("metrics", ["accuracy", "f1"])),
            "verbose": params.get("verbose", False),
        }
```

Usage in YAML:
```yaml
task: my_custom_eval
executor: slurm_swedev2

params:
  mode: lenient
  metrics: [precision, recall, f1]
  verbose: true
  # ... other standard params
```

### Example 3: Custom Template for Radical Changes

Create `devrun/templates/my_custom_template.sh.j2`:
```jinja2
#!/bin/bash
set -x

# Custom environment setup
module load python/3.11
source /opt/custom_env/bin/activate

# Custom logic
export CUSTOM_VAR={{ custom_value | shell_quote }}

python {{ script | shell_quote }} \
    --custom-flag \
    --input {{ input_path | shell_quote }}
```

Usage in YAML:
```yaml
task: swe_bench_agentic
executor: slurm_swedev2

params:
  template: "my_custom_template.sh.j2"
  run_script: "tools/run_eval.py"
  custom_value: "my_value"
  input_path: "/data/input"
```

## Parameter Reference

### `run_script`
- **Type:** `str`
- **Default:** `"benchmarks/swebench/run_infer.py"`
- **Description:** Path to the Python script to execute.

### `script_args`
- **Type:** `dict[str, Any]`
- **Default:** `{}`
- **Description:** Additional CLI arguments to pass to the script.
  - Keys are argument names (underscores converted to hyphens).
  - `True` values produce bare flags (`--flag`).
  - `False` values are omitted.
  - Other values produce `--key value` pairs with shell quoting.

### `template`
- **Type:** `str`
- **Default:** `"swe_bench_agentic.sh.j2"`
- **Description:** Jinja2 template name or path. Templates in `devrun/templates/` can be referenced by name. Absolute paths are also supported.

### `extra_flags`
- **Type:** `list[str]`
- **Default:** `["--use-legacy-tools", "--bind-dev-sdk"]`
- **Description:** Raw flag strings appended to the command (before `script_args`).

## Hook Methods for Subclassing

### `_get_run_script(params: dict[str, Any]) -> str`
Returns the script path. Default reads `params.get("run_script", "benchmarks/swebench/run_infer.py")`.

### `_get_default_flags(params: dict[str, Any]) -> list[str]`
Returns extra CLI flags. Default reads `params.get("extra_flags", ["--use-legacy-tools", "--bind-dev-sdk"])`.

### `_get_script_args(params: dict[str, Any]) -> dict[str, Any]`
Returns a dict of script arguments. Default reads `params.get("script_args", {})`.

**Priority:** When a param key matches a hook method's purpose (e.g., `script_args` in params), the param value is used directly and the hook is not called.

## Shell Quoting and Safety

All `script_args` values are automatically shell-quoted via `shlex.quote()` in the template. Argument names with underscores are converted to hyphens (e.g., `batch_size` → `--batch-size`).
