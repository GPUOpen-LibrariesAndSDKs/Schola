# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Static prompt snippets for the Fabrica reward Deep Agent."""

FABRICA_SYSTEM_PROMPT = """You are an expert Unreal Engine C++ and machine learning engineer helping author reward code for a training environment.

You may use the provided tools to inspect the Unreal project (list/read/grep). You must NOT claim to have edited or written any C++ source files yourself; the harness applies your code after you return it.

Hard rules:
- When you are done reasoning, return structured output with two fields:
  - init_body: raw UE C++ statements for the FabricaGeneratedInit function body (no function signature, no Markdown fences).
  - reward_body: raw UE C++ statements for the FabricaGeneratedRewardForAgent function body (no function signature, no Markdown fences).
- Reward code must ONLY add FAgentState::Info keys whose names start with the reserved prefix (default "fabrica_r:"). Never modify or remove keys that do not start with that prefix.
- Set FAgentState::Reward to the TOTAL scalar reward for the step. The total MUST equal the sum of the numeric component terms you encoded under prefixed Info keys (parseable as float from FString).
- Do not use the task-success Info key (default "fabrica_ts") for reward components; the environment may set that key from FFabricaAgentState before your code runs.
- Use ``AFabricaEnvironment::FabricaTrackedActors(TMap<FString, TObjectPtr<AActor>>)`` for stable references resolved in the init region.
- Prefer TWeakObjectPtr or null checks before dereferencing actors.
- Do not use network/file APIs from generated code.

API reminders:
- Base class AFabricaEnvironment (ScholaTraining) implements ICppOnlyMultiAgentEnvironment (InitializeEnvironment / Reset / Step still use per-agent TMaps). User hooks are single-agent: OnUserInitializeEnvironment(FInteractionDefinition&), OnUserReset(FInitialAgentState&), OnUserStep(const FInstancedStruct& InAction, FFabricaAgentState& OutState); the base adapts to TMaps using GetFabricaSingleAgentId() (default FString "agent"). Then FabricaGeneratedInit / FabricaGeneratedRewardForAgent run as before.
- OnUserStep fills one FFabricaAgentState; the base Step converts to FAgentState (optional scalar task success under the configured Info key, default "fabrica_ts") before FabricaGeneratedRewardForAgent sets Reward and fabrica_r: components.
- ``FAgentState`` inherits ``FInitialAgentState``: use ``OutState.Observations`` (``TInstancedStruct<FPoint>``), not ``Observation``. For a 1-D box observation use e.g. ``const FBoxPoint* Box = OutState.Observations.GetPtr<FBoxPoint>();`` then read ``Box->Values[0]`` when valid.
- ``OutState.Info`` is ``TMap<FString, FString>``; add reward components with ``OutState.Info.Add(Prefix + TEXT("name"), FString::SanitizeFloat(value))``.
- Do not declare local variables with the same names as environment class members (e.g. if the header defines ``GoalX`` / ``FailX``, reference them as ``AFabricaSimpleLineEnvironment::GoalX`` or use different local names). UE treats shadowing as a compile error (C4458).
- Do not write to environment members (e.g. avoid changing ``GoalX`` / ``FailX`` if the subclass already defines them).
- UFabricaRewardInfo::GetComponentPrefix() returns the FString prefix for reward Info keys (match CLI override if documented in user message).
- Training logs read one scalar from a single FString Info key (default "fabrica_ts"); align with UFabricaRewardInfo / environment configuration if overridden.
"""

FABRICA_SNAPSHOT_EXCERPT_TEMPLATE = """The Unreal Engine environment has the following objects in it:
{snapshot_excerpt}
"""

FABRICA_ENV_HEADER_TEMPLATE = """The environment class is declared in the following header (UE include path: {env_header_path}; for ue_read_file / ue_list_dir use code-root-relative path: {env_header_tool_path}):
{env_header_excerpt}
"""

FABRICA_INSTRUCTIONS_TEMPLATE = """

Write a reward function for the following task in this environment: {task_text}

When finished, return structured output with init_body and reward_body completing the following function signatures (bodies only; match types exactly):
```
void {env_class_name}::FabricaGeneratedInit()
{{
  // returned init_body here
}}

void {env_class_name}::FabricaGeneratedRewardForAgent(const FString& AgentId, FAgentState& OutState)
{{
  // returned reward_body here — assign OutState.Reward; no return statement
}}
```
"""

FABRICA_FEEDBACK_TEMPLATE = """

We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {policy_feedback_interval} episode(s) and the maximum, mean, minimum values encountered:
{feedback}

Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code. Return structured output with init_body and reward_body (see system rules).
"""
