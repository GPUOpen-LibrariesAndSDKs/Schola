// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Boolean vector observation/action. Mirrors Unreal FMultiBinaryPoint / Proto MultiBinaryPoint.
    /// </summary>
    public sealed class MultiBinaryPoint : Point
    {
        public List<bool> Values { get; set; } = new List<bool>();
    }
}
