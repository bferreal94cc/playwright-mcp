# Google Agents CLI Setup

This document describes the setup and configuration of Google Agents CLI (`google-agents-cli`) for this Playwright MCP project.

## Setup Summary

Google Agents CLI has been installed globally using `uvx` (Python's universal package installer). This provides access to the Agent Development Kit (ADK) for building, evaluating, and deploying AI agents.

### Installation Details

**Installed**: August 11, 2026  
**Method**: `uvx google-agents-cli setup`  
**Installation Path**: `~/.local/share/uv/tools/google-agents-cli/`  
**Version**: 1.0+ (latest from PyPI via uvx)

### Installed Skills

The setup command automatically installed 7 skills for coding agents:

1. **google-agents-cli-adk-code** - ADK Python API patterns and agent code writing
2. **google-agents-cli-deploy** - Deployment to Cloud Run, GKE, or Agent Runtime
3. **google-agents-cli-eval** - Evaluation methodology and dataset management
4. **google-agents-cli-observability** - Cloud Trace, logging, and monitoring setup
5. **google-agents-cli-publish** - Publishing agents to Gemini Enterprise
6. **google-agents-cli-scaffold** - Project scaffolding and enhancement
7. **google-agents-cli-workflow** - End-to-end ADK development lifecycle

All skills are symlinked to Claude Code (`~/.agents/skills/google-agents-cli-*/`).

### Security Assessment

All installed skills passed security risk assessment:
- **Gen**: Safe (all)
- **Socket**: 0 alerts (all)
- **Snyk**: Low-to-Medium risk (expected for Python packages)

See https://skills.sh/google/agents-cli for detailed security information.

## Usage

### Available Commands

```bash
# Create a new agent project
uvx google-agents-cli create my-agent

# Start the local playground
uvx google-agents-cli playground

# Evaluate agents
uvx google-agents-cli eval generate  # Run inference
uvx google-agents-cli eval grade     # Grade traces

# Scaffold/enhance projects
uvx google-agents-cli scaffold enhance .

# Deploy agents
uvx google-agents-cli deploy

# Publish agents
uvx google-agents-cli publish gemini-enterprise

# Check project info
uvx google-agents-cli info
```

For complete help, run:
```bash
uvx google-agents-cli --help
```

## Integration with Playwright MCP

The Google Agents CLI can be used to:

1. **Create ADK agents** that use Playwright MCP for browser automation
2. **Evaluate agent capabilities** with test datasets
3. **Deploy agents** with observability and monitoring
4. **Scaffold projects** with CI/CD and deployment infrastructure

### Example Integration Pattern

ADK tools are plain Python functions passed to the agent via `tools=[...]`. The
docstring and type hints are what the model sees, so they carry the description
of when and how to use the tool. This mirrors the agent the CLI itself
scaffolds (`agents-cli create` → `app/agent.py`):

```python
from google.adk.agents import Agent
from google.adk.models import Gemini

MODEL = "gemini-3.6-flash"


def navigate_and_interact(url: str) -> str:
    """Navigate to a URL and report what is on the page.

    Args:
        url: The absolute URL to open.

    Returns:
        A text summary of the resulting page.
    """
    # Drive the Playwright MCP server here (e.g. an MCP client session
    # calling browser_navigate followed by browser_snapshot).
    ...


root_agent = Agent(
    name="root_agent",
    model=Gemini(model=MODEL),
    instruction="You are a browser automation assistant.",
    tools=[navigate_and_interact],
)
```

To connect the Playwright MCP server to an agent directly rather than wrapping
it by hand, use ADK's MCP tool support — see the
[ADK tools documentation](https://google.github.io/adk-docs/tools/) for the
current API.

## Authentication

To fully use Google Agents CLI features (especially deployment and publishing):

```bash
uvx google-agents-cli login -i
```

The `-i`/`--interactive` flag is required — `login` refuses to run without it.
To check the current state instead of authenticating, use
`uvx google-agents-cli login --status`.

This enables:
- Deployment to Google Cloud projects
- Publishing to Gemini Enterprise
- Accessing Agent Runtime services

## Next Steps

1. **Create a project**: Use `/google-agents-cli-scaffold` skill or `agents-cli scaffold create`
2. **Write agent code**: Use `/google-agents-cli-adk-code` skill for ADK patterns
3. **Set up evaluation**: Use `/google-agents-cli-eval` skill for eval datasets
4. **Deploy**: Use `/google-agents-cli-deploy` skill when ready
5. **Monitor**: Use `/google-agents-cli-observability` skill for production monitoring

## References

- [Google Agents CLI Docs](https://cloud.google.com/agents/docs)
- [Agent Development Kit (ADK)](https://cloud.google.com/agents/docs/agent-development-kit)
- [Playwright MCP](https://github.com/microsoft/playwright-mcp)
