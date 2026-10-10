// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Continuous float vector observation/action. Mirrors Unreal FBoxPoint / Proto BoxPoint.
    /// </summary>
    public sealed class BoxPoint : Point
    {
        public List<float> Values { get; set; } = new List<float>();

        /// <summary>
        /// Multi-dimensional shape. Empty means 1D with length <see cref="Values"/>.Count.
        /// </summary>
        public List<int> Shape { get; set; } = new List<int>();
    }
}
