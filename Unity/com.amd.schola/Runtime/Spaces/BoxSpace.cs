// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Continuous box space. Mirrors Unreal FBoxSpace / Proto BoxSpace.
    /// </summary>
    public sealed class BoxSpace : Space
    {
        public List<BoxSpaceDimension> Dimensions { get; set; } = new List<BoxSpaceDimension>();

        /// <summary>
        /// Multi-dimensional shape. Empty means 1D with length <see cref="Dimensions"/>.Count.
        /// </summary>
        public List<int> Shape { get; set; } = new List<int>();
    }
}
