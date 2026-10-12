// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola {
    /// <summary>
    /// Represents categorical choices from 0 to high-1
    /// </summary>
    public sealed class DiscreteSpace : Space {
        public int High { get; set; }

        public DiscreteSpace(){}

        public DiscreteSpace(int high) {
            High = high;
        }
    }
}
