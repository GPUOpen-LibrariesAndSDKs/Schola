// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola {
    /// <summary>
    /// Observation and action spaces a policy maps between.
    /// </summary>
    public sealed class InteractionDefinition {
        public Space ObservationSpace { get; set; }
        public Space ActionSpace { get; set; }

        public InteractionDefinition(){}

        public InteractionDefinition(Space observationSpace, Space actionSpace) {
            ObservationSpace = observationSpace;
            ActionSpace = actionSpace;
        }
    }
}
