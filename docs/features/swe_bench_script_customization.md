# SWE-bench Script Customization

This document describes how to customize the SWE-bench agentic task's Python script invocation using the new `script_args` parameter and hook methods.

## Overview

The `SWEBenchAgenticTask` now supports three levels of customization:

1. **Parameter-based**: Pass `script_args` dict in your task config
2. **Task inheritance**: Override `_get_script_args()` hook method in a subclass
3. **Custom templates**: Provide a completely custom Jinja2 template

## Method 1: Parameter-based customization

The simplest approach is to add `script_args` to your task YAML config.

### Example: Adding custom CLI arguments

```yaml
task: swe_bench_agentic
executor: slurm_pascal
params:
  dataset: /data/SWE-bench_Verified
  llm_config: configs/gpt4.json
  script_args:
    batch_size: 32
    temperature: 0.7
    enable_cache: true
```

This generates:
```bash
python run_infer.py ... --batch-size 32 --temperature 0.7 --enable-cache
```

### Argument name transformation

- Underscores in keys are converted to hyphens: `batch_size` → `--batch-size`
- Boolean `true` produces a bare flag: `enable_cache: true` → `--enable-cache`
- Boolean `false` omits the argument entirely
- All other values are rendered as `--key value` with proper shell quoting

### Shell safety

Values are automatically shell-quoted to prevent injection:
```yaml
script_args:
  output_path: "/tmp/test dir/output"  # → --output-path '/tmp/test dir/output'
  pattern: "*.py"                      # → --pattern '*.py'
```

## Method 2: Task inheritance

For reusable patterns, override the `_get_script_args()` hook method.

### Example: Custom evaluation task

```python
from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask
from typing import Any

class CustomEvalTask(SWEBenchAgenticTask):
    def _get_script_args(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "evaluator_mode": params.get("mode", "strict"),
            "enable_cache": True,
            "metrics": ",".join(params.get("metrics", ["accuracy"])),
        }
```

Use in config:
```yaml
task: custom_eval_task
executor: slurm_pascal
params:
  dataset: /data/SWE-bench_Verified
  llm_config: configs/gpt4.json
  mode: lenient
  metrics: [f1, recall, precision]
```

### Merging params with defaults

```python
class TaskWithDefaults(SWEBenchAgenticTask):
    def _get_script_args(self, params: dict[str, Any]) -> dict[str, Any]:
        # Start with defaults
        args = {
            "enable_logging": True,
            "cache_dir": "/tmp/cache"
        }
        # Merge in user-provided script_args
        args.update(params.get("script_args", {}))
        return args
```

## Method 3: Custom templates

For radical customizations, provide a custom Jinja2 template.

### Example: Custom template file

Create `my_custom_template.sh.j2`:
```jinja2
#!/bin/bash
set -x

echo "Custom execution logic"

python {{ script | shell_quote }} \
    --dataset {{ dataset | shell_quote }} \
    --custom-flag{% for arg_name, arg_value in script_args.items() %} \
    --{{ arg_name | replace('_', '-') }} {{ arg_value | string | shell_quote }}{% endfor %}

echo "Done"
```

Use in config:
```yaml
task: swe_bench_agentic
executor: slurm_pascal
params:
  dataset: /data/SWE-bench_Verified
  llm_config: configs/gpt4.json
  template: /path/to/my_custom_template.sh.j2
  script_args:
    batch_size: 32
```

### Template requirements

- Use absolute paths for custom templates
- Apply `| shell_quote` filter to all user-provided values
- Use `| string` before `| shell_quote` for numeric values
- Available variables: see `devrun/templates/swe_bench_agentic.sh.j2` header comment

## Backward Compatibility

All changes are backward compatible:
- Existing configs work without modification
- `script_args` parameter is optional (defaults to empty dict)
- Default template unchanged when `template` parameter not provided
- Hook methods only called when not overridden in params

## Implementation Details

### Hook method signature

```python
def _get_script_args(self, params: dict[str, Any]) -> dict[str, Any]:
    """Override in subclasses to provide custom script arguments.
    
    Returns a dict mapping argument names to values:
      {"arg_name": "value"}  → --arg-name value
      {"flag": True}         → --flag
      {"flag": False}        → (omitted)
    
    Argument names with underscores are converted to hyphens automatically.
    """
    return params.get("script_args", {})
```

### Template rendering

The default implementation passes `script_args` to the template context:
```python
command = render_template(
    template_name,
    # ... other context ...
    script_args=script_args,
)
```

Templates iterate over `script_args.items()` and render arguments with proper shell quoting and boolean flag handling.
