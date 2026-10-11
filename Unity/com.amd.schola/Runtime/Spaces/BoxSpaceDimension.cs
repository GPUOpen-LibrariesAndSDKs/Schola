// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola
{
    /// <summary>
    /// Bounds for one continuous dimension. Null low/high means unbounded on that side.
    /// </summary>
    public sealed class BoxSpaceDimension
    {
        public float? Low { get; set; }
        public float? High { get; set; }

        public BoxSpaceDimension()
        {
        }

        public BoxSpaceDimension(float? low, float? high)
        {
            Low = low;
            High = high;
        }
    }
}
