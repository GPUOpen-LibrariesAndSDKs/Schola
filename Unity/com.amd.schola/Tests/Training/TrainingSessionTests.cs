// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AMD.Schola.Training;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Training
{
    public sealed class TrainingSessionTests
    {
        const string SingleAgent = ScholaEnvironment.DefaultAgentId;

        sealed class Environment : IEnvironment
        {
            public int Initializes,
                Steps,
                Resets;
            public float Position;
            public bool Terminate;
            public bool Truncate;
            public int? LastSeed;
            public IReadOnlyDictionary<string, string> Options;

            public InteractionDefinition Initialize()
            {
                Initializes++;
                return new InteractionDefinition(
                    new BoxSpace { Dimensions = { new BoxSpaceDimension() } },
                    new BoxSpace { Dimensions = { new BoxSpaceDimension(-1, 1) } }
                );
            }

            public void Seed(int seed) => LastSeed = seed;

            public void SetOptions(IReadOnlyDictionary<string, string> options) =>
                Options = options;

            public InitialAgentState Reset()
            {
                Resets++;
                Position = 0;
                return new InitialAgentState
                {
                    Observations = new BoxPoint { Values = { Position } },
                };
            }

            public AgentState Step(Point action)
            {
                Steps++;
                Position += ((BoxPoint)action).Values[0];
                return new AgentState
                {
                    Observations = new BoxPoint { Values = { Position } },
                    Reward = 2,
                    Terminated = Terminate,
                    Truncated = Truncate,
                    Info = { ["step"] = Steps.ToString() },
                };
            }
        }

        /// <summary>
        /// Two agents with different action bounds. Follows the
        /// <see cref="IMultiAgentEnvironment"/> finished-agent rule: omit the action to
        /// keep the terminal state; send an action to step again.
        /// </summary>
        sealed class TeamEnvironment : IMultiAgentEnvironment
        {
            public const string A = "agentA";
            public const string B = "agentB";
            static readonly string[] s_Ids = { A, B };

            public int Initializes,
                Steps,
                Resets;
            public readonly Dictionary<string, float> Position = new Dictionary<string, float>();
            public readonly Dictionary<string, float> Reward = new Dictionary<string, float>
            {
                [A] = 1,
                [B] = 2,
            };
            public readonly HashSet<string> Terminate = new HashSet<string>();
            public readonly HashSet<string> Truncate = new HashSet<string>();
            public string OmitFromStep;
            public IReadOnlyDictionary<string, Point> LastActions;
            readonly Dictionary<string, AgentState> m_Finished =
                new Dictionary<string, AgentState>();

            public IReadOnlyDictionary<string, InteractionDefinition> Initialize()
            {
                Initializes++;
                return new Dictionary<string, InteractionDefinition>
                {
                    [A] = Definition(1),
                    [B] = Definition(2),
                };
            }

            static InteractionDefinition Definition(float bound) =>
                new InteractionDefinition(
                    new BoxSpace { Dimensions = { new BoxSpaceDimension() } },
                    new BoxSpace { Dimensions = { new BoxSpaceDimension(-bound, bound) } }
                );

            public void Seed(int seed) { }

            public void SetOptions(IReadOnlyDictionary<string, string> options) { }

            public IReadOnlyDictionary<string, InitialAgentState> Reset()
            {
                Resets++;
                m_Finished.Clear();
                var initial = new Dictionary<string, InitialAgentState>();
                foreach (var id in s_Ids)
                {
                    Position[id] = 0;
                    initial[id] = new InitialAgentState
                    {
                        Observations = new BoxPoint { Values = { 0 } },
                    };
                }
                return initial;
            }

            public IReadOnlyDictionary<string, AgentState> Step(
                IReadOnlyDictionary<string, Point> actions
            )
            {
                Steps++;
                LastActions = actions;
                var states = new Dictionary<string, AgentState>();
                foreach (var id in s_Ids)
                {
                    var hasAction = actions.TryGetValue(id, out var action) && action != null;
                    if (m_Finished.TryGetValue(id, out var terminal) && !hasAction)
                    {
                        states[id] = terminal.Copy();
                        continue;
                    }
                    m_Finished.Remove(id);
                    Position[id] += ((BoxPoint)actions[id]).Values[0];
                    var state = new AgentState
                    {
                        Observations = new BoxPoint { Values = { Position[id] } },
                        Reward = Reward[id],
                        Terminated = Terminate.Contains(id),
                        Truncated = Truncate.Contains(id),
                    };
                    states[id] = state;
                    if (state.Terminated || state.Truncated)
                        m_Finished[id] = state.Copy();
                }
                if (OmitFromStep != null)
                    states.Remove(OmitFromStep);
                return states;
            }
        }

        static TrainingRequest Request(string command) => new TrainingRequest { command = command };

        static TrainingAgentAction Action(string id, float value) =>
            new TrainingAgentAction
            {
                id = id,
                action = TrainingWire.EncodePoint(new BoxPoint { Values = { value } }),
            };

        static TrainingRequest Step(float value = 0.5f) =>
            new TrainingRequest
            {
                command = "step",
                actions = new[] { Action(SingleAgent, value) },
            };

        static TrainingRequest TeamStep(params TrainingAgentAction[] actions) =>
            new TrainingRequest { command = "step", actions = actions };

        static TrainingRequest TeamStep() =>
            TeamStep(Action(TeamEnvironment.A, 0.5f), Action(TeamEnvironment.B, 1.5f));

        static TrainingAgentState State(TrainingAgentState[] states, string id) =>
            states.Single(state => state.id == id);

        static float Observation(TrainingAgentState state) =>
            ((BoxPoint)TrainingWire.DecodePoint(state.observations)).Values[0];

        static TrainingSession StartedTeam(TeamEnvironment env, int autoreset = 0)
        {
            var session = new TrainingSession(env);
            Assert.That(
                session.Handle(new TrainingRequest { command = "start", autoreset = autoreset }).ok,
                Is.True
            );
            Assert.That(session.Handle(Request("reset")).ok, Is.True);
            return session;
        }

        [Test]
        public void RequiresStartupAndReset_AndForwardsSeedAndOptions()
        {
            var env = new Environment();
            var session = new TrainingSession(env);
            Assert.That(session.Handle(Step()).ok, Is.False);
            Assert.That(session.Handle(Request("start")).ok, Is.True);
            Assert.That(session.Handle(Step()).ok, Is.False);
            Assert.That(
                session
                    .Handle(
                        new TrainingRequest
                        {
                            command = "reset",
                            hasSeed = true,
                            seed = 17,
                            options = new[]
                            {
                                new TrainingEntry { key = "difficulty", value = "easy" },
                            },
                        }
                    )
                    .ok,
                Is.True
            );
            Assert.That(env.LastSeed, Is.EqualTo(17));
            Assert.That(env.Options["difficulty"], Is.EqualTo("easy"));
            var states = session.Handle(Step()).states;
            Assert.That(states, Has.Length.EqualTo(1));
            Assert.That(states[0].id, Is.EqualTo(SingleAgent));
            Assert.That(states[0].reward, Is.EqualTo(2));
            Assert.That(states[0].info.Single().key, Is.EqualTo("step"));
        }

        [Test]
        public void SingleAgentEnvironment_IsDefinedUnderTheSingleAgentId()
        {
            var session = new TrainingSession(new Environment());
            session.Handle(Request("start"));
            var definition = session.Handle(Request("definition"));
            Assert.That(definition.ok, Is.True, definition.error);
            Assert.That(definition.agents, Has.Length.EqualTo(1));
            Assert.That(definition.agents[0].id, Is.EqualTo(SingleAgent));
            Assert.That(definition.agents[0].actionSpace[0].dimensions[0].high, Is.EqualTo(1));
            var reset = session.Handle(Request("reset"));
            Assert.That(reset.states.Single().id, Is.EqualTo(SingleAgent));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void SameStepReset_PreservesFinalObservationAndEpisodeFlags(bool truncate)
        {
            var env = new Environment { Terminate = !truncate, Truncate = truncate };
            var session = new TrainingSession(env);
            session.Handle(Request("start"));
            session.Handle(Request("reset"));
            var response = session.Handle(Step());
            Assert.That(response.ok, Is.True, response.error);
            var state = response.states.Single();
            Assert.That(state.terminated, Is.EqualTo(!truncate));
            Assert.That(state.truncated, Is.EqualTo(truncate));
            Assert.That(Observation(state), Is.EqualTo(0.5f));
            Assert.That(response.hasInitial, Is.True);
            Assert.That(response.initialStates.Single().id, Is.EqualTo(SingleAgent));
            Assert.That(Observation(response.initialStates.Single()), Is.Zero);
            Assert.That(env.Resets, Is.EqualTo(2));
        }

        [TestCase(1)]
        [TestCase(2)]
        public void NextStepAndDisabledReset_RespectEpisodeBoundary(int mode)
        {
            var env = new Environment { Terminate = true };
            var session = new TrainingSession(env);
            session.Handle(new TrainingRequest { command = "start", autoreset = mode });
            session.Handle(Request("reset"));
            Assert.That(session.Handle(Step()).hasInitial, Is.False);
            var next = session.Handle(Step());
            Assert.That(env.Steps, Is.EqualTo(1));
            Assert.That(next.ok, Is.EqualTo(mode == 1));
            if (mode == 1)
            {
                Assert.That(next.states.Single().reward, Is.Zero);
                Assert.That(next.states.Single().terminated, Is.False);
                Assert.That(env.Resets, Is.EqualTo(2));
            }
        }

        [Test]
        public void InvalidAction_DoesNotAdvanceEnvironment_AndRequiresReset()
        {
            var env = new Environment();
            var session = new TrainingSession(env);
            session.Handle(Request("start"));
            session.Handle(Request("reset"));
            Assert.That(session.Handle(Step(2)).ok, Is.False);
            Assert.That(env.Steps, Is.Zero);
            Assert.That(session.Handle(Step()).ok, Is.False);
            session.Handle(Request("reset"));
            Assert.That(session.Handle(Step()).ok, Is.True);
        }

        [Test]
        public void Reconnect_DoesNotReinitializeResetSnapshot()
        {
            var env = new Environment();
            var session = new TrainingSession(env);
            session.Handle(Request("start"));
            session.Handle(Request("reset"));
            session.Handle(Step());
            session.Handle(Request("close"));
            Assert.That(session.Handle(Request("start")).ok, Is.True);
            Assert.That(env.Initializes, Is.EqualTo(1));
            Assert.That(session.Handle(Step()).ok, Is.False);
            session.Handle(Request("reset"));
            Assert.That(env.Position, Is.Zero);
        }

        [Test]
        public void UnsupportedVersion_IsRejectedNamingBothVersions()
        {
            var session = new TrainingSession(new Environment());
            var response = session.Handle(new TrainingRequest { version = 1, command = "start" });
            Assert.That(response.ok, Is.False);
            Assert.That(response.version, Is.EqualTo(TrainingWire.Version));
            Assert.That(response.error, Does.Contain("version 1"));
            Assert.That(response.error, Does.Contain($"version {TrainingWire.Version}"));
        }

        [Test]
        public void MultiAgent_DefinitionAndReset_ListEveryAgentWithItsOwnSpaces()
        {
            var env = new TeamEnvironment();
            var session = new TrainingSession(env);
            session.Handle(Request("start"));
            var definition = session.Handle(Request("definition"));
            Assert.That(definition.ok, Is.True, definition.error);
            Assert.That(
                definition.agents.Select(agent => agent.id),
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            Assert.That(
                definition
                    .agents.Single(agent => agent.id == TeamEnvironment.A)
                    .actionSpace[0]
                    .dimensions[0]
                    .high,
                Is.EqualTo(1)
            );
            Assert.That(
                definition
                    .agents.Single(agent => agent.id == TeamEnvironment.B)
                    .actionSpace[0]
                    .dimensions[0]
                    .high,
                Is.EqualTo(2)
            );
            var reset = session.Handle(Request("reset"));
            Assert.That(reset.ok, Is.True, reset.error);
            Assert.That(
                reset.states.Select(state => state.id),
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            Assert.That(reset.states.All(state => Observation(state) == 0), Is.True);
            Assert.That(env.Initializes, Is.EqualTo(1));
        }

        [Test]
        public void MultiAgent_Step_ReturnsEveryAgentAndChecksEachActionAgainstItsOwnSpace()
        {
            var env = new TeamEnvironment();
            var session = StartedTeam(env);
            // 1.5 is inside agent B's bounds and outside agent A's.
            var response = session.Handle(TeamStep());
            Assert.That(response.ok, Is.True, response.error);
            Assert.That(response.hasInitial, Is.False);
            Assert.That(State(response.states, TeamEnvironment.A).reward, Is.EqualTo(1));
            Assert.That(State(response.states, TeamEnvironment.B).reward, Is.EqualTo(2));
            Assert.That(Observation(State(response.states, TeamEnvironment.A)), Is.EqualTo(0.5f));
            Assert.That(Observation(State(response.states, TeamEnvironment.B)), Is.EqualTo(1.5f));

            var rejected = session.Handle(
                TeamStep(Action(TeamEnvironment.A, 1.5f), Action(TeamEnvironment.B, 1.5f))
            );
            Assert.That(rejected.ok, Is.False);
            Assert.That(env.Steps, Is.EqualTo(1));
        }

        [TestCase("unknown")]
        [TestCase("duplicate")]
        [TestCase("missing")]
        [TestCase("none")]
        public void MultiAgent_BadAgentIds_DoNotAdvanceEnvironment_AndRequireReset(string problem)
        {
            var env = new TeamEnvironment();
            var session = StartedTeam(env);
            var a = Action(TeamEnvironment.A, 0.5f);
            var b = Action(TeamEnvironment.B, 0.5f);
            TrainingAgentAction[] actions;
            switch (problem)
            {
                case "unknown":
                    actions = new[] { a, b, Action("other", 0.5f) };
                    break;
                case "duplicate":
                    actions = new[] { a, a, b };
                    break;
                case "missing":
                    actions = new[] { a };
                    break;
                default:
                    actions = null;
                    break;
            }

            var response = session.Handle(TeamStep(actions));

            Assert.That(response.ok, Is.False);
            Assert.That(env.Steps, Is.Zero);
            Assert.That(session.Handle(TeamStep()).ok, Is.False);
            session.Handle(Request("reset"));
            Assert.That(session.Handle(TeamStep()).ok, Is.True);
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        public void MultiAgent_OneFinishedAgent_DoesNotEndTheEnvironment(int mode)
        {
            var env = new TeamEnvironment();
            env.Terminate.Add(TeamEnvironment.A);
            var session = StartedTeam(env, mode);

            var first = session.Handle(TeamStep());
            Assert.That(first.ok, Is.True, first.error);
            Assert.That(State(first.states, TeamEnvironment.A).terminated, Is.True);
            Assert.That(State(first.states, TeamEnvironment.B).terminated, Is.False);
            Assert.That(first.hasInitial, Is.False);
            Assert.That(env.Resets, Is.EqualTo(1));

            // The finished agent needs no action; the live one keeps stepping.
            var second = session.Handle(TeamStep(Action(TeamEnvironment.B, 0.25f)));
            Assert.That(second.ok, Is.True, second.error);
            Assert.That(env.Steps, Is.EqualTo(2));
            Assert.That(env.LastActions.Keys, Is.EquivalentTo(new[] { TeamEnvironment.B }));
            var finished = State(second.states, TeamEnvironment.A);
            Assert.That(finished.terminated, Is.True);
            Assert.That(finished.reward, Is.EqualTo(1));
            Assert.That(Observation(finished), Is.EqualTo(0.5f));
            Assert.That(Observation(State(second.states, TeamEnvironment.B)), Is.EqualTo(1.75f));

            // An invalid action for a finished agent is rejected (it would revive and step).
            var invalid = session.Handle(
                TeamStep(Action(TeamEnvironment.A, 99), Action(TeamEnvironment.B, 0.25f))
            );
            Assert.That(invalid.ok, Is.False);
            Assert.That(env.Steps, Is.EqualTo(2));

            session.Handle(Request("reset"));
            env.Terminate.Add(TeamEnvironment.A);
            Assert.That(session.Handle(TeamStep()).ok, Is.True);
            env.Terminate.Remove(TeamEnvironment.A);

            // A valid action for a finished agent steps it again.
            var revived = session.Handle(
                TeamStep(Action(TeamEnvironment.A, 0.5f), Action(TeamEnvironment.B, 0.25f))
            );
            Assert.That(revived.ok, Is.True, revived.error);
            Assert.That(
                env.LastActions.Keys,
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            Assert.That(State(revived.states, TeamEnvironment.A).terminated, Is.False);
            Assert.That(Observation(State(revived.states, TeamEnvironment.A)), Is.EqualTo(1f));
        }

        [Test]
        public void MultiAgent_SameStepReset_WaitsForTheLastAgentThenReturnsEveryInitialState()
        {
            var env = new TeamEnvironment();
            env.Terminate.Add(TeamEnvironment.A);
            var session = StartedTeam(env);
            Assert.That(session.Handle(TeamStep()).hasInitial, Is.False);
            env.Truncate.Add(TeamEnvironment.B);

            var last = session.Handle(TeamStep(Action(TeamEnvironment.B, 0.25f)));

            Assert.That(last.ok, Is.True, last.error);
            Assert.That(State(last.states, TeamEnvironment.A).terminated, Is.True);
            Assert.That(State(last.states, TeamEnvironment.B).truncated, Is.True);
            Assert.That(State(last.states, TeamEnvironment.B).terminated, Is.False);
            Assert.That(Observation(State(last.states, TeamEnvironment.B)), Is.EqualTo(1.75f));
            Assert.That(last.hasInitial, Is.True);
            Assert.That(
                last.initialStates.Select(state => state.id),
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            Assert.That(last.initialStates.All(state => Observation(state) == 0), Is.True);
            Assert.That(env.Resets, Is.EqualTo(2));

            // The new episode has every agent live again.
            env.Terminate.Clear();
            env.Truncate.Clear();
            Assert.That(session.Handle(TeamStep(Action(TeamEnvironment.B, 0.25f))).ok, Is.False);
            session.Handle(Request("reset"));
            Assert.That(session.Handle(TeamStep()).ok, Is.True);
        }

        [Test]
        public void MultiAgent_NextStepReset_ReturnsEveryAgentWithZeroRewardAndClearedFlags()
        {
            var env = new TeamEnvironment();
            env.Terminate.Add(TeamEnvironment.A);
            env.Truncate.Add(TeamEnvironment.B);
            var session = StartedTeam(env, 1);
            var final = session.Handle(TeamStep());
            Assert.That(final.hasInitial, Is.False);
            Assert.That(env.Resets, Is.EqualTo(1));

            // NEXT_STEP ignores this step's actions, so none are needed.
            var reset = session.Handle(TeamStep(null));

            Assert.That(reset.ok, Is.True, reset.error);
            Assert.That(env.Steps, Is.EqualTo(1));
            Assert.That(env.Resets, Is.EqualTo(2));
            Assert.That(reset.hasInitial, Is.False);
            Assert.That(
                reset.states.Select(state => state.id),
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            foreach (var state in reset.states)
            {
                Assert.That(state.reward, Is.Zero);
                Assert.That(state.terminated, Is.False);
                Assert.That(state.truncated, Is.False);
                Assert.That(Observation(state), Is.Zero);
            }
            env.Terminate.Clear();
            env.Truncate.Clear();
            Assert.That(session.Handle(TeamStep()).ok, Is.True);
            Assert.That(env.Steps, Is.EqualTo(2));
        }

        [Test]
        public void MultiAgent_DisabledReset_RejectsStepsOnceEveryAgentFinished()
        {
            var env = new TeamEnvironment();
            env.Terminate.Add(TeamEnvironment.A);
            env.Terminate.Add(TeamEnvironment.B);
            var session = StartedTeam(env, 2);
            Assert.That(session.Handle(TeamStep()).ok, Is.True);

            Assert.That(session.Handle(TeamStep()).ok, Is.False);

            Assert.That(env.Steps, Is.EqualTo(1));
            Assert.That(env.Resets, Is.EqualTo(1));
            env.Terminate.Clear();
            session.Handle(Request("reset"));
            Assert.That(session.Handle(TeamStep()).ok, Is.True);
        }

        [Test]
        public void MultiAgent_NonFiniteReward_IsRejectedNamingTheAgent()
        {
            var env = new TeamEnvironment();
            env.Reward[TeamEnvironment.B] = float.NaN;
            var session = StartedTeam(env);

            var response = session.Handle(TeamStep());

            Assert.That(response.ok, Is.False);
            Assert.That(response.error, Does.Contain(TeamEnvironment.B));
            Assert.That(session.Handle(TeamStep()).ok, Is.False);
        }

        [Test]
        public void MultiAgent_EnvironmentOmittingAnAgentState_IsRejected()
        {
            var env = new TeamEnvironment { OmitFromStep = TeamEnvironment.B };
            var session = StartedTeam(env);

            var response = session.Handle(TeamStep());

            Assert.That(response.ok, Is.False);
            Assert.That(response.states, Is.Null.Or.Empty);
        }

        [Test]
        public void Wire_RoundTripsNestedDictAndRejectsDuplicateOrMalformedNodes()
        {
            var source = new DictPoint
            {
                Points =
                {
                    ["nested"] = new DictPoint
                    {
                        Points = { ["move"] = new BoxPoint { Values = { 1, 2, 3 } } },
                    },
                    ["choice"] = new DiscretePoint { Value = 2 },
                },
            };
            var nodes = TrainingWire.EncodePoint(source);
            var decoded = (DictPoint)TrainingWire.DecodePoint(nodes);
            Assert.That(((DiscretePoint)decoded.Points["choice"]).Value, Is.EqualTo(2));
            Assert.That(
                ((BoxPoint)((DictPoint)decoded.Points["nested"]).Points["move"]).Values,
                Is.EqualTo(new[] { 1f, 2f, 3f })
            );
            // Also exercise the actual Unity serializer used on the socket boundary.
            var request = JsonUtility.FromJson<TrainingRequest>(
                JsonUtility.ToJson(
                    new TrainingRequest
                    {
                        actions = new[]
                        {
                            new TrainingAgentAction { id = "agent", action = nodes },
                        },
                    }
                )
            );
            Assert.That(request.actions.Single().id, Is.EqualTo("agent"));
            Assert.That(
                TrainingWire.DecodePoint(request.actions.Single().action),
                Is.TypeOf<DictPoint>()
            );
            nodes[2].parent = 100;
            Assert.Throws<ArgumentException>(() => TrainingWire.DecodePoint(nodes));
            Assert.Throws<ArgumentException>(() =>
                TrainingWire.DecodePoint(
                    new[]
                    {
                        new TrainingNode { kind = "box", values = new[] { float.NaN } },
                    }
                )
            );
        }

        [Test]
        public void Wire_RoundTripsPerAgentResponseThroughJsonUtility()
        {
            var env = new TeamEnvironment();
            env.Terminate.Add(TeamEnvironment.A);
            env.Terminate.Add(TeamEnvironment.B);
            var session = StartedTeam(env);
            var definition = JsonUtility.FromJson<TrainingResponse>(
                JsonUtility.ToJson(session.Handle(Request("definition")))
            );
            var step = JsonUtility.FromJson<TrainingResponse>(
                JsonUtility.ToJson(session.Handle(TeamStep()))
            );

            Assert.That(definition.version, Is.EqualTo(TrainingWire.Version));
            Assert.That(
                definition.agents.Select(agent => agent.id),
                Is.EquivalentTo(new[] { TeamEnvironment.A, TeamEnvironment.B })
            );
            Assert.That(
                definition.agents.All(agent => agent.observationSpace[0].kind == "box"),
                Is.True
            );
            Assert.That(step.ok, Is.True, step.error);
            Assert.That(State(step.states, TeamEnvironment.B).reward, Is.EqualTo(2));
            Assert.That(State(step.states, TeamEnvironment.B).terminated, Is.True);
            Assert.That(Observation(State(step.states, TeamEnvironment.B)), Is.EqualTo(1.5f));
            Assert.That(step.hasInitial, Is.True);
            Assert.That(Observation(State(step.initialStates, TeamEnvironment.A)), Is.Zero);
        }

        [Test]
        public void BoundedTranslateActuator_PublishesAndEnforcesBounds()
        {
            var go = new GameObject("BoundedActuator");
            try
            {
                var actuator = go.AddComponent<TranslateActuator>();
                actuator.MaxDisplacement = 0.1f;
                Assert.That(
                    ((BoxSpace)actuator.GetActionSpace()).Dimensions[0].High,
                    Is.EqualTo(0.1f)
                );
                Assert.Throws<ArgumentException>(() =>
                    actuator.TakeAction(new BoxPoint { Values = { 1, 0, 0 } })
                );
                Assert.That(go.transform.position, Is.EqualTo(Vector3.zero));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }
    }
}
