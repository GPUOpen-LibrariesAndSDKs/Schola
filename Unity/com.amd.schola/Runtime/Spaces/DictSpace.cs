// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Named map of spaces. Mirrors Unreal FDictSpace / Proto DictSpace.
    /// </summary>
    public sealed class DictSpace : Space
    {
        public Dictionary<string, Space> Spaces { get; set; } = new Dictionary<string, Space>();
    }
}
