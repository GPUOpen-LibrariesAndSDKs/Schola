# Godot demo

This standalone Godot .NET project contains the Deliverable 1 Schola environment demo.

## Requirements

- Godot 4.7.2 with .NET support
- .NET 8 SDK

## Run the demo

1. Open Godot and choose **Import** (or **Open**) in the Project Manager.
2. Select `Godot/demo/project.godot` from this repository. The project root is `Godot/demo`.
3. Wait for Godot to import the assets, then build the C# project with the **Build** button in the editor.
4. Press **F6** to run the current demo scene, or **F5** to run the configured main scene.

The main scene is `assets/Demo.tscn`, and the Schola addon is included under `addons/schola` and enabled in the project settings.

## Controls

- **W/A/S/D**: move the player
- **Enter**: add a test reward
- **Esc**: fail the current episode

Move the player to the green goal and stay within its radius to complete the episode. The environment resets after completion or failure; the reward status panel shows feedback.
