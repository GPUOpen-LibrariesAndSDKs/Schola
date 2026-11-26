# Lyra StateTree NPC Demo

This demo adds StateTree to Lyra NPCs while keeping their existing Behavior Tree AI.

## What's Included

- `BP_AMD_LyraBot`: A custom Lyra Pawn with a StateTree Component.
- `ST_AMD_LyraBot`: Patrol and Chase states, controlled by the `Should Chase` parameter.
- Conditional transitions between Patrol and Chase.

## Setup

1. Copy the `Plugins` folder from this directory into your Lyra project.
2. Open `HeroData_ShooterGame` in ShooterCore.
3. Temporarily change its Pawn Class to `BP_AMD_LyraBot`.
4. Open `L_ShooterGym` and press Play.
5. Check the Output Log for `BP_AMD_LyraBot`.

Restore the original Pawn Class after testing, since this change affects both players and bots.

For macOS setup, see [Lyra Integration](../../../../Docs/Lyra-Integration.md).

## Verification

Tested on macOS ARM64 with Unreal Engine 5.8.4.

Multiple Lyra Pawn instances successfully started StateTree:

    [BP_AMD_LyraBot_C_0] Running
    [BP_AMD_LyraBot_C_1] Running

## Next Steps

- Verify Patrol/Chase transitions at runtime.
- Connect StateTree states to Shared Policy head selection.
- Integrate Architecture C inference with Lyra gameplay.
