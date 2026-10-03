# Schola for Godot - Team 20, The Hard Workers

Schola for Godot is a CSC301 project developed by our 7-person student team in partnership with AMD. The project is currently in its planning and prototyping stage.

## Partner introduction

Our partner is AMD, the organization that maintains the open-source Schola reinforcement-learning toolkit.

- **Primary contact:** Alexander Cann, Member of Technical Staff, [alexander.cann@amd.com](mailto:alexander.cann@amd.com)
- **Secondary contact:** TianYue "Michael" Liu, Senior Software Engineer, [tianyliu@amd.com](mailto:tianyliu@amd.com)

## Project description

We are building a Godot 4.7 port of AMD Schola. It will let Godot developers define reinforcement-learning environments and agents, train them with Schola's Python tools, export trained policies to ONNX, and run those policies inside Godot without Python. The port removes the need for developers to create their own engine-to-Python communication, episode coordination, and inference infrastructure.

## Key features

- **Godot-native environment definition:** Developers will configure environments and agents through Godot nodes and the Inspector.
- **Observation and action spaces:** The port will support Box, Discrete, MultiDiscrete, and MultiBinary spaces and validate values against their declared spaces.
- **Python training integration:** Godot environments will communicate with Schola's existing Python ecosystem through its gRPC protocol.
- **Complete episode lifecycle:** The integration will coordinate actions, observations, rewards, terminal states, truncation, and resets across complete episodes.
- **Local ONNX inference:** Exported policies will run inside Godot without a live Python process or training connection.
- **Separable packaging:** Training-only dependencies will be removable from exported games that only need inference.

The full MVP and its acceptance criteria are in the [D1 planning document](deliverables/D1/planning.md). The [architecture diagram](deliverables/D1/d1-architecture-diagram.png) shows the planned components and workflow.

## Instructions

The Godot add-on is not yet runnable because the project is in its planning and prototyping stage. Once the first implementation is available, this section will explain how to install the add-on, create an environment and agent, connect to the Python training tools, export a policy, and run that policy in Godot.

## Development requirements

The planned engine-side implementation targets Godot 4.7 and uses a native C++ GDExtension. The training integration will use gRPC and Protocol Buffers, while shipped-policy inference will use ONNX Runtime. The Python side will reuse Schola's Gymnasium, Stable-Baselines3, and RLlib integrations where practical. Exact setup and build commands will be added after the prototype establishes the final dependency layout.

## Deployment and GitHub workflow

This project is a developer library rather than a hosted service. The Godot integration will be distributed as an add-on, with training-only code packaged separately from the core and inference components.

The 7 team members track work on the [Trello board](https://trello.com/b/Ry0Qkx2R). Each change is developed on a branch and submitted to `main` through a pull request. At least 1 other team member must review and approve the pull request before it is merged. The related Trello card remains in progress until the pull request is merged. This workflow keeps work attributable, reduces conflicts, and prevents unreviewed changes from entering `main`.

Commit messages follow Conventional Commits, using prefixes such as `feat:`, `fix:`, `docs:`, and `test:`. Changes that affect shared architecture or protocol behavior are discussed with the team and AMD before implementation.

## Coding standards and guidelines

C++ code follows Godot's C++ style and is formatted with `clang-format`. Python code follows PEP 8 and is formatted with Black. New behavior must include relevant tests, and generated Protocol Buffer files must not be edited by hand.

## License

The project uses the [MIT License](LICENSE.txt), matching AMD Schola. This permits use, modification, and redistribution while requiring preservation of the license and copyright notice.

## Project resources

- [D1 planning document](deliverables/D1/planning.md)
- [D1 architecture diagram](deliverables/D1/d1-architecture-diagram.png)
- [Team and stakeholder records](deliverables/team/)
- [Meeting minutes](deliverables/team/minutes/)
- [Trello project board](https://trello.com/b/Ry0Qkx2R)

## Deployed URL and access instructions

There is no deployed application for D1. The finished project will be used as a local Godot add-on, with Python packages installed locally for training. Access and installation instructions will be added when a runnable version is available.

## D3 improvement highlight

Not applicable for D1. This section will summarize the changes made between D2 and D3.
