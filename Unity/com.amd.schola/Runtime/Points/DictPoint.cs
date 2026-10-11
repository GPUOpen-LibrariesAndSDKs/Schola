// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Named map of points. Mirrors Unreal FDictPoint / Proto DictPoint.
    /// </summary>
    public sealed class DictPoint : Point
    {
        public Dictionary<string, Point> Points { get; set; } = new Dictionary<string, Point>();
    }
}
