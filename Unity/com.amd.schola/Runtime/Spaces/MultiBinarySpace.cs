// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola
{
    /// <summary>
    /// Independent boolean dimensions. Mirrors Unreal FMultiBinarySpace / Proto MultiBinarySpace.
    /// </summary>
    public sealed class MultiBinarySpace : Space
    {
        /// <summary>Number of binary dimensions (Gymnasium MultiBinary n).</summary>
        public int Shape { get; set; }

        public MultiBinarySpace() { }

        public MultiBinarySpace(int shape)
        {
            Shape = shape;
        }
    }
}
