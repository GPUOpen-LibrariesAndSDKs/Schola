# Lyra StateTree NPC Demo

This demo adds StateTree to Lyra NPCs while keeping their existing Behavior Tree AI.

## What's Included

- `BP_AMD_LyraBot`: A custom Lyra Pawn with a StateTree Component.
- `ST_AMD_LyraBot`: Patrol and Chase states using actor-level `ShouldChase` with bidirectional On Tick transitions.
- `STT_AMD_LogPatrol` and `STT_AMD_LogChase`: Blueprint tasks that log state entry events.

## Setup

1. Copy all four .uasset files (BP_AMD_LyraBot, ST_AMD_LyraBot, STT_AMD_LogPatrol, and STT_AMD_LogChase) into LyraStarterGame/Plugins/GameFeatures/ShooterCore/Content/Game/AMD/. Create the AMD folder if needed.
2. Open `HeroData_ShooterGame` in ShooterCore.
3. Temporarily change its Pawn Class to `BP_AMD_LyraBot`.
4. Open `L_ShooterGym` and press Play.
5. Check the Output Log for `BP_AMD_LyraBot`.

Restore the original Pawn Class after testing, since this change affects both players and bots.

For macOS setup, see [Lyra Integration](../../../Docs/Lyra-Integration.md)

## Verification

Tested on macOS ARM64 with Unreal Engine 5.8.4.

Multiple Lyra Pawn instances successfully started StateTree:

    [BP_AMD_LyraBot_C_0] Running
    [BP_AMD_LyraBot_C_1] Running

## Runtime Transition Verification

Tested in `L_ShooterGym` on macOS ARM64 with UE 5.8.4.

- Multiple `BP_AMD_LyraBot` instances ran successfully.
- Actor-level `ShouldChase` toggled approximately every 3 seconds.
- Patrol and Chase EnterState logging tasks executed at runtime.
- Logs showed alternating `AMD ENTER PATROL` and `AMD ENTER CHASE` events.
- Per-NPC independent state selection has not yet been verified with actor-specific state logs.

## Next Steps

- Add per-NPC identifiers to verify independent state selection across NPC instances.
- Connect StateTree states to Shared Policy head selection.
- Integrate Architecture C inference with Lyra gameplay.
