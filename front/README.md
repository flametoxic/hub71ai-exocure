# CURE Desktop

React/Electron frontend for the sibling `../backend` FastAPI server.
See [the workspace README](../README.md) for setup, launch, configuration and integration tests.

From this directory, `npm.cmd start` builds and launches the desktop;
`npm.cmd run test:integration` checks the real desktop/backend integration.
The desktop starts `../backend/tools/run_demo.py`.
The managed server stores resident data in `../backend/core_store` by default.

The UI opens fullscreen; F11 toggles fullscreen. Phone and Home hub share resident
state. The insights panel is collapsible. Bruno Ace is bundled with its OFL license.
