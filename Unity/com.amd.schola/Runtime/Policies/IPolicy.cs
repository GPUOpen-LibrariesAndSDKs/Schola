// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola {
    /// <summary>
    /// Policy maps observations to actions.
    /// </summary>
    public interface IPolicy {
        /// <summary>
        /// Prepares the policy for the given observation and action spaces.
        /// </summary>
        /// <returns>True if the policy can produce actions for this definition.</returns>
        bool Init(InteractionDefinition definition);

        /// <summary>
        /// Produces actions from observations. Call <see cref="Init"/> first.
        /// </summary>
        /// <returns>True if inference succeeded.</returns>
        bool Think(Point observations, out Point actions);
    }
}
